import { describe, expect, it } from "vitest";
import * as THREE from "three";
import { CpuShader, matrixValue } from "./glsl.js";
import { CpuRasterizer, createBuckets, depthPass } from "./rasterizer.js";
import { CpuEnvironment } from "./environment.js";
import { serializeScene, serializeTexture } from "./sceneSerialization.js";
import type { FrameSettings } from "./frameSettings.js";
import { sampleTexture } from "./texture.js";
import { encodeColor, toneMap, toneMap3 } from "./color.js";
import { CpuMaterial } from "./material.js";
import { ShadowBvh } from "./bvh.js";

const context = { texture: () => [1, .5, .25, 1] };

const settings: FrameSettings = {
  renderScale: 1,
  maxSamples: 1,
  tileSize: 16,
  shadows: true,
  environmentIntensity: 1,
  environmentRotation: 0,
  backgroundMode: "transparent",
  backgroundColor: "#000000",
  dofEnabled: false,
  dofFocusDistance: 3,
  dofAperture: 2.8,
  bokehBlades: 0,
  tonemapping: "linear",
  exposure: 1,
};

const environment = new CpuEnvironment();

function triangle(color: number, z = -2) {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute([-1, -1, z, 1, -1, z, 0, 1, z], 3));
  geometry.setAttribute("uv", new THREE.Float32BufferAttribute([0, 0, 1, 0, .5, 1], 2));

  return new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({ color }));
}

async function render(
  meshes: THREE.Object3D[],
  options: Partial<FrameSettings> = {},
  width = 32,
  height = 32,
  configure?: (scene: THREE.Scene) => void) {
  const scene = new THREE.Scene();
  scene.add(...meshes);
  configure?.(scene);
  const { scene: serialized } = await serializeScene(scene);
  const camera = new THREE.PerspectiveCamera(60, width / height, .1, 100);
  camera.updateMatrixWorld();

  const renderer = new CpuRasterizer(serialized, {
    matrixWorld: camera.matrixWorld.toArray(),
    fov: 60,
    near: .1,
    far: 100,
  }, {
    ...settings,
    ...options,
  }, width, height, environment);

  for (const _step of renderer.prepare()) { /* drain cooperative preparation */ }

  const image = new Uint8ClampedArray(width * height * 4);

  for (const bucket of renderer.buckets) {
    const job = renderer.renderBucket(bucket);
    let step = job.next();
    while (!step.done)
      step = job.next();
    for (let y = 0; y < bucket.height; y++)
      image.set(
        step.value.pixels.subarray(
          y * bucket.width * 4,
          (y + 1) * bucket.width * 4,
        ),
        ((bucket.y + y) * width + bucket.x) * 4,
      );
  }

  return {
    image,
    pixel: (x = 16, y = 16) => Array.from(image.subarray(
      (y * width + x) * 4,
      (y * width + x) * 4 + 4,
    )),
  };
}

describe("CPU GLSL execution", () => {
  it("renders a transported custom shader using macros, structs, arrays and overloads", async() => {
    const mesh = triangle(0xffffff);

    const material = new THREE.ShaderMaterial({
      vertexShader: "void main(){gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0);}",
      fragmentShader: `
#define APPLY(c, g) ((c) * (g))
struct Tint { vec3 color; mat3 basis; };
uniform Tint tint;
vec3 shade(float x) { return vec3(x); }
vec3 shade(vec3 x) { return APPLY(tint.basis*x, 0.5); }
void main(){
  vec3 colors[2]=vec3[](vec3(0.0),tint.color);
  gl_FragColor=vec4(shade(colors[1]),1.0);
}`,
      uniforms: { tint: { value: { color: new THREE.Vector3(1, 0, 0), basis: new THREE.Matrix3() } } },
    });

    const custom = new THREE.Mesh(mesh.geometry, material);
    const result = await render([custom]);
    expect(result.pixel()[0]).toBeGreaterThan(150);
    expect(result.pixel().slice(1)).toEqual([0, 0, 255]);
  });
  it("runs vertex matrices, constructors and varying assignments", () => {
    const shader = new CpuShader("varying vec2 vUv; void main(){ vUv=uv; gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0); }");

    const result = shader.run({

      uv: [.25, .75],
      position: [1, 2, 3],
      gl_Position: [0, 0, 0, 0],
      projectionMatrix: matrixValue(new THREE.Matrix4().toArray()),
      modelViewMatrix: matrixValue(new THREE.Matrix4().makeTranslation(
        4,
        0,
        0,
      ).toArray()),
    }, context)!;

    expect(result.gl_Position).toEqual([5, 2, 3, 1]);
    expect(result.vUv).toEqual([.25, .75]);
  });
  it("runs functions, loops, branching, swizzles and texture sampling", () => {
    const shader = new CpuShader(`uniform sampler2D tex;
    float gain(float x){return x*2.0;}
    void main(){vec4 c=texture2D(tex,vec2(0.5));for(int i=0;i<3;i++){if(i==1)continue;c.rgb*=0.5;}c.r=gain(c.r);gl_FragColor=c;}`);

    expect(shader.run(
      {
        tex: { texture: "test" },
        gl_FragColor: [0, 0, 0, 0],
      },
      context,
    )!.gl_FragColor).toEqual([.5, .125, .0625, 1]);
  });
  it("supports conditional compilation and matrix uniforms", () => {
    const shader = new CpuShader(
      "#ifdef COLOR\nuniform mat3 m;\n#endif\nvoid main(){\n#if defined(COLOR) && COLOR > 1\ngl_FragColor=vec4(m*vec3(1.0),1.0);\n#else\ndiscard;\n#endif\n}",
      { COLOR: 2 },
    );

    expect(shader.run(
      {
        m: [
          2,
          0,
          0,
          0,
          3,
          0,
          0,
          0,
          4,
        ],
        gl_FragColor: [0, 0, 0, 0],
      },
      context,
    )!.gl_FragColor).toEqual([2, 3, 4, 1]);
  });
  it("preserves inout values and local scopes", () => {
    const shader = new CpuShader("void doubleValue(inout vec3 c){c*=2.0;} void main(){vec3 c=vec3(0.25);{vec3 c=vec3(0.0);}doubleValue(c);gl_FragColor=vec4(c,1.0);}");
    expect(shader.run({ gl_FragColor: [0, 0, 0, 0] }, context)!.gl_FragColor).toEqual([.5, .5, .5, 1]);
  });
  it("supports discard and enforces a finite instruction budget", () => {
    expect(new CpuShader("void main(){discard;}").run({}, context)).toBeNull();
    expect(() => new CpuShader("void main(){while(true){}} ").run(
      {},
      {
        ...context,
        budget: 100,
      },
    )).toThrow("budget exceeded");
    expect(() =>
      new CpuShader("void main(){gl_FragColor=vec4(dFdx(1.0));}").run(
        { gl_FragColor: [0, 0, 0, 0] },
        context,
      )).toThrow("fragment quad");
    expect(() => new CpuShader("void main(){globalThis.fetch();}")).toThrow();
  });
});
describe("CPU scene transport", () => {
  it("copies buffers, preserves material classes, draw ranges and parent visibility", async() => {
    const a = triangle(0xff0000), b = triangle(0xffffff);
    const parent = new THREE.Group();
    parent.visible = false;
    parent.add(b);
    a.geometry.setDrawRange(0, 3);
    const scene = new THREE.Scene();
    scene.add(a, parent);
    const result = await serializeScene(scene);
    expect(result.scene.meshes).toHaveLength(1);
    expect(result.scene.meshes[0].materials[0].type).toBe("MeshBasicMaterial");
    expect(result.scene.meshes[0].attributes.position.data.buffer).not.toBe(a.geometry.getAttribute("position").array.buffer);
    structuredClone(result.scene, { transfer: result.transfer as ArrayBuffer[] });
    expect(a.geometry.getAttribute("position").array.byteLength).toBeGreaterThan(0);
  });
  it("exports instances with unique world transforms and deduplicates transfers", async() => {
    const mesh = triangle(0xffffff), instances = new THREE.InstancedMesh(mesh.geometry, mesh.material, 2);
    instances.setMatrixAt(1, new THREE.Matrix4().makeTranslation(3, 0, 0));
    const scene = new THREE.Scene();
    scene.add(instances);
    const result = await serializeScene(scene);
    expect(result.scene.meshes).toHaveLength(2);
    expect(result.scene.meshes[1].matrixWorld[12]).toBe(3);
    expect(new Set(result.transfer).size).toBe(result.transfer.length);
  });
  it("preserves floating point textures, alpha and UV transforms", async() => {
    const texture = new THREE.DataTexture(
      new Float32Array([4, .25, .5, .75]),
      1,
      1,
      THREE.RGBAFormat,
      THREE.FloatType,
    );

    texture.repeat.set(2, 3);
    const serialized = await serializeTexture(texture);
    expect(Array.from(serialized.data)).toEqual([4, .25, .5, .75]);
    expect(serialized.matrix[0]).toBe(2);
    expect(sampleTexture(serialized, [.5, .5])).toEqual([4, .25, .5, .75]);
  });
  it("fails explicitly for unsupported callbacks instead of swapping materials", async() => {
    const mesh = triangle(0xffffff);

    mesh.material.onBeforeCompile = (_shader, renderer) => {
      renderer.getSize(new THREE.Vector2());
    };

    const scene = new THREE.Scene();
    scene.add(mesh);
    await expect(serializeScene(scene)).rejects.toThrow("onBeforeCompile");
  });
});
describe("CPU rasterization", () => {
  it("covers odd image dimensions exactly once and starts at the center", () => {
    const buckets = createBuckets(35, 29, 16), covered = new Uint8Array(35 * 29);
    for (const b of buckets)
      for (let y = b.y; y < b.y + b.height; y++)
        for (let x = b.x; x < b.x + b.width; x++)
          covered[y * 35 + x]++;
    expect([...covered].every(v => v === 1)).toBe(true);
    expect(buckets[0].x).toBe(16);
  });
  it("renders unlit material colors and transparent background", async() => {
    const result = await render([triangle(0xff0000)]);
    expect(result.pixel()).toEqual([255, 0, 0, 255]);
    expect(result.pixel(0, 0)).toEqual([0, 0, 0, 0]);
  });
  it("resolves occlusion independently of mesh order", async() => {
    for (const meshes of [[triangle(
      0xff0000,
      -2,
    ), triangle(
      0x0000ff,
      -3,
    )], [triangle(
      0x0000ff,
      -3,
    ), triangle(
      0xff0000,
      -2,
    )]])
      expect((await render(meshes)).pixel()).toEqual([255, 0, 0, 255]);
  });
  it("clips triangles crossing the near plane without exploding bounds", async() => {
    const mesh = triangle(0x00ff00);
    mesh.geometry.getAttribute("position").setZ(2, .2);
    const result = await render([mesh]);
    expect(result.image.some((v, i) => i % 4 === 3 && v === 255)).toBe(true);
  });
  it("rejects geometry behind the camera and honors back-face culling", async() => {
    expect((await render([triangle(0xff0000, 2)])).image.every(v => v === 0)).toBe(true);
    const mesh = triangle(0xff0000);
    mesh.geometry.setIndex([0, 2, 1]);
    expect((await render([mesh])).pixel()[3]).toBe(0);
    mesh.material.side = THREE.DoubleSide;
    expect((await render([mesh])).pixel()[3]).toBe(255);
  });
  it("sorts transparency per pixel and preserves straight-alpha PNG output", async() => {
    const near = triangle(0xff0000, -2);
    near.material.transparent = true;
    near.material.opacity = .5;
    expect((await render([near])).pixel()).toEqual([255, 0, 0, 128]);
    const result = await render([near, triangle(0x0000ff, -3)]);
    expect(result.pixel()).toEqual([188, 0, 188, 255]);
  });
  it("executes custom vertex/fragment shaders and interpolates varyings", async() => {
    const mesh = triangle(0xffffff);

    const custom = new THREE.Mesh(
      mesh.geometry,
      new THREE.ShaderMaterial({
        vertexShader: "varying vec2 vUv;void main(){vUv=uv;gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0);}",
        fragmentShader: "varying vec2 vUv;void main(){gl_FragColor=vec4(vUv,0.0,1.0);}",
      }),
    );

    const result = await render([custom]);
    expect(result.pixel()[0]).toBeGreaterThan(150);
    expect(result.pixel()[1]).toBeGreaterThan(150);
    expect(result.pixel()[2]).toBe(0);
  });
  it("applies alpha testing before writing depth", async() => {
    const near = triangle(0xff0000, -2);
    near.material.opacity = .1;
    near.material.alphaTest = .5;
    expect((await render([near, triangle(0x0000ff, -3)])).pixel()).toEqual([0, 0, 255, 255]);
  });
  it("produces the same image regardless of bucket size", async() => {
    const a = await render(
        [triangle(0xff9933)],
        {
          tileSize: 16,
          maxSamples: 4,
        },
        37,
        31,
      ), b = await render(
        [triangle(0xff9933)],
        {
          tileSize: 32,
          maxSamples: 4,
        },
        37,
        31,
      );

    expect(a.image).toEqual(b.image);
  });
  it("does not double-blend shared edges", async() => {
    const mesh = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), new THREE.MeshBasicMaterial({

      color: 0xff0000,
      transparent: true,
      opacity: .5,
      depthWrite: false,
    }));

    mesh.position.z = -2;
    const image = await render([mesh]);
    expect(image.pixel(16, 16)[3]).toBe(128);
  });
  it("accepts empty scenes and rejects unbounded allocations", async() => {
    expect((await render([])).image.every(v => v === 0)).toBe(true);
    expect(() => new CpuRasterizer({

      meshes: [],
      lights: [],
      textures: [],
    }, {

      matrixWorld: new THREE.Matrix4().toArray(),
      fov: 60,
      near: .1,
      far: 100,
    }, settings, 100000, 100000, environment)).toThrow("resolution");
  });
  it("honors configurable depth comparisons", () => {
    expect(depthPass(0, .1, .5)).toBe(false);
    expect(depthPass(1, .9, .1)).toBe(true);
    expect(depthPass(3, .5, .5)).toBe(true);
    expect(depthPass(2, .5, .5)).toBe(false);
  });
});
describe("lighting and color", () => {
  it("queries shadows through the BVH and respects alpha rejection", () => {
    const bvh = new ShadowBvh([{
      positions: [[-1, -1, -2], [1, -1, -2], [0, 1, -2]],
      castShadow: true,
    }]);

    expect(bvh.occluded([0, 0, 0], [0, 0, -1], 3, -1, () => true)).toBe(true);
    expect(bvh.occluded([0, 0, 0], [0, 0, -1], 1, -1, () => true)).toBe(false);
    expect(bvh.occluded([0, 0, 0], [0, 0, -1], 3, -1, () => false)).toBe(false);
  });
  it.each(["aces", "agx", "neutral", "linear"] as const)(
    "maps HDR into finite display values with %s",
    mode => {
      const values = toneMap([10, .2, .01], 1, mode);
      expect(values.every(v => Number.isFinite(v) && v >= 0 && v <= 1)).toBe(true);
      expect(toneMap([10, 10, 10], 0, mode).every(v => Math.abs(v) < .001)).toBe(true);
    });
  it("resolves the pixel with the same ACES curve as the array version", () => {
    for (const rgb of [[10, .2, .01], [0, 0, 0], [.5, .5, .5], [2, -1, 100]]) {
      const [r, g, b] = rgb;
      expect(toneMap3(r, g, b, 1.3, "aces")).toEqual(toneMap(rgb, 1.3, "aces"));
      expect(toneMap3(r, g, b, .7, "agx")).toEqual(toneMap(rgb, .7, "agx"));
    }
  });
  it("keeps sRGB textures as bytes and decodes to linear when sampling", async() => {
    const texture = new THREE.DataTexture(
      new Uint8Array([128, 255, 0, 255]), 1, 1, THREE.RGBAFormat, THREE.UnsignedByteType);

    texture.colorSpace = THREE.SRGBColorSpace;
    const serialized = await serializeTexture(texture);
    expect(serialized.colorSpace).toBe(THREE.SRGBColorSpace);
    expect(serialized.data).toBeInstanceOf(Uint8Array);
    expect(serialized.data.byteLength).toBe(4);
    const pixel = sampleTexture(serialized, [.5, .5]);
    expect(pixel[0]).toBeCloseTo(((128 / 255 + .055) / 1.055) ** 2.4, 6);
    expect(pixel.slice(1)).toEqual([1, 0, 1]);
  });
  it("marks as ray-opaque only materials that never let light through", async() => {
    const opaque = new THREE.MeshStandardMaterial({ color: 0xff0000 });
    const cutout = new THREE.MeshStandardMaterial({ color: 0xff0000, alphaTest: .5 });
    const glass = new THREE.MeshPhysicalMaterial({ transmission: 1 });
    const faded = new THREE.MeshBasicMaterial({ transparent: true, opacity: .5 });
    const scene = new THREE.Scene();
    for (const material of [opaque, cutout, glass, faded]) scene.add(new THREE.Mesh(triangle(0).geometry, material));
    const { scene: serialized } = await serializeScene(scene);
    const flags = serialized.meshes.map(mesh => new CpuMaterial(mesh.materials[0], new Map()).rayOpaque);
    expect(flags).toEqual([true, false, false, false]);
  });
  it("lights inside the spot cone and not outside it", async() => {
    const lit = () => {
      const spot = new THREE.SpotLight(0xffffff, 30, 0, Math.PI / 8, .2, 0);
      spot.position.set(0, 0, 1);

      return spot;
    };

    const inside = lit(), outside = lit();
    inside.target.position.set(0, 0, -2);
    outside.target.position.set(5, 0, 1);
    const surface = () => new THREE.Mesh(triangle(0).geometry, new THREE.MeshLambertMaterial({ color: 0xffffff }));
    // No environment: only the spot light contributes.
    const on = await render([surface(), inside, inside.target], { environmentIntensity: 0 });
    const off = await render([surface(), outside, outside.target], { environmentIntensity: 0 });
    expect(on.pixel()[0]).toBeGreaterThan(100);
    expect(off.pixel()[0]).toBeLessThan(10);
  });
  it.each([
    THREE.MeshLambertMaterial,
    THREE.MeshPhongMaterial,
    THREE.MeshToonMaterial,
    THREE.MeshStandardMaterial,
    THREE.MeshPhysicalMaterial,
  ])(
    "shades native %s with a light",
    async(Material) => {
      const base = triangle(0xffffff), mesh = new THREE.Mesh(base.geometry, new Material({ color: 0xff0000 }));
      const light = new THREE.DirectionalLight(0xffffff, 3);
      light.position.set(0, 0, 1);
      const result = await render([mesh, light]);
      expect(result.pixel()[0]).toBeGreaterThan(100);
      expect(result.pixel()[3]).toBe(255);
    });
});

describe("native shader hooks", () => {
  it("captures data-only onBeforeCompile material inputs and final color", async() => {
    const mesh = triangle(0xffffff);

    mesh.material.onBeforeCompile = shader => {
      shader.fragmentShader
        = shader.fragmentShader.replace(
          "#include <color_fragment>",
          "#include <color_fragment>\ndiffuseColor.rgb = vec3(1.0, 0.0, 0.0);",
        );
      shader.fragmentShader
        = shader.fragmentShader.replace(
          "#include <dithering_fragment>",
          "#include <dithering_fragment>\ngl_FragColor.rgb = gl_FragColor.bgr;",
        );
    };

    expect((await render([mesh])).pixel()).toEqual([0, 0, 255, 255]);
  });
  it("captures custom vertex displacement and fragment discard", async() => {
    const mesh = triangle(0xffffff);

    mesh.material.onBeforeCompile = shader => {
      shader.vertexShader
        = shader.vertexShader.replace(
          "#include <begin_vertex>",
          "#include <begin_vertex>\ntransformed.x += 10.0;",
        );
    };

    expect((await render([mesh])).pixel()[3]).toBe(0);

    mesh.material.onBeforeCompile = shader => {
      shader.fragmentShader
        = shader.fragmentShader.replace(
          "#include <clipping_planes_fragment>",
          "#include <clipping_planes_fragment>\ndiscard;",
        );
    };

    expect((await render([mesh])).pixel()[3]).toBe(0);
  });
  it("keeps the focal plane sharp with thin-lens sampling", async() => {
    const mesh = triangle(0xffaa11, -3);
    const sharp = await render([mesh], { maxSamples: 8 });

    const focused = await render(
      [mesh],
      {
        maxSamples: 8,
        dofEnabled: true,
        dofFocusDistance: 3,
        dofAperture: .5,
      },
    );

    expect(focused.image).toEqual(sharp.image);
  });
});

describe("Three parity", () => {
  it("passes ambient irradiance through the Lambert BRDF", async() => {
    const mesh = new THREE.Mesh(triangle(0xffffff).geometry, new THREE.MeshLambertMaterial({ color: 0xffffff }));
    const full = await render([mesh, new THREE.AmbientLight(0xffffff, Math.PI)], { environmentIntensity: 0 });
    expect(full.pixel()).toEqual([255, 255, 255, 255]);
    const unit = await render([mesh, new THREE.AmbientLight(0xffffff, 1)], { environmentIntensity: 0 });
    expect(unit.pixel()[0]).toBeGreaterThan(145);
    expect(unit.pixel()[0]).toBeLessThan(160);
  });
  it("keeps specular and emissive over transmitted light and tints it with the base color", async() => {
    const back = triangle(0xffffff, -3);

    const glass = new THREE.Mesh(triangle(0xffffff).geometry,
      new THREE.MeshPhysicalMaterial({ color: 0x00ff00, transmission: 1, roughness: 0 }));

    const tinted = await render([glass, back], { environmentIntensity: 0 });
    expect(tinted.pixel()[1]).toBeGreaterThan(200);
    expect(tinted.pixel()[0]).toBeLessThan(40);
    glass.material.emissive = new THREE.Color(0x0000ff);
    const glowing = await render([glass, back], { environmentIntensity: 0 });
    expect(glowing.pixel()[1]).toBeGreaterThan(200);
    expect(glowing.pixel()[2]).toBeGreaterThan(200);
  });
  it("bends transmitted light by ior and thickness, like getIBLVolumeRefraction", async() => {
    // The red/green seam sits at pixel 23.5: pixel 24 sees green straight through and red once refracted inwards.
    const half = (color: number, x: number) => {
      const mesh = new THREE.Mesh(new THREE.PlaneGeometry(6, 10), new THREE.MeshBasicMaterial({ color }));
      mesh.position.set(x, 0, -3);

      return mesh;
    };

    const backdrop = () => [half(0xff0000, .812 - 3), half(0x00ff00, .812 + 3)];

    const pane = (thickness: number) => {
      const mesh = new THREE.Mesh(new THREE.PlaneGeometry(10, 10),
        new THREE.MeshPhysicalMaterial({ color: 0xffffff, transmission: 1, roughness: 0, ior: 1.5, thickness }));

      mesh.position.z = -2;

      return mesh;
    };

    const flat = (await render([pane(0), ...backdrop()], { environmentIntensity: 0 })).pixel(24, 16);
    expect(flat[0]).toBeLessThan(40);
    expect(flat[1]).toBeGreaterThan(200);
    const thick = (await render([pane(4), ...backdrop()], { environmentIntensity: 0 })).pixel(24, 16);
    expect(thick[0]).toBeGreaterThan(200);
    expect(thick[1]).toBeLessThan(40);
  });
  it("splits the transmitted channels by ior with dispersion", async() => {
    const half = (color: number, x: number) => {
      const mesh = new THREE.Mesh(new THREE.PlaneGeometry(6, 10), new THREE.MeshBasicMaterial({ color }));
      mesh.position.set(x, 0, -3);

      return mesh;
    };

    const pane = new THREE.Mesh(new THREE.PlaneGeometry(10, 10), new THREE.MeshPhysicalMaterial({
      color: 0xffffff, transmission: 1, roughness: 0, ior: 1.5, thickness: 4, dispersion: 40 }));

    pane.position.z = -2;
    // Red refracts with ior 1 (straight through, onto the green half); green with 1.5 (onto the red half).
    const split = await render([pane, half(0xff0000, .812 - 3), half(0x00ff00, .812 + 3)], { environmentIntensity: 0 });
    expect(split.pixel(24, 16)[0]).toBeLessThan(40);
    expect(split.pixel(24, 16)[1]).toBeLessThan(40);
  });
  it("lets transmissive casters block direct light, like Three's shadow maps", async() => {
    const floor = new THREE.Mesh(new THREE.PlaneGeometry(6, 6), new THREE.MeshLambertMaterial({ color: 0xffffff }));
    floor.position.z = -3;
    floor.receiveShadow = true;
    const glass = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), new THREE.MeshPhysicalMaterial({ transmission: 1 }));
    glass.position.z = -2;
    glass.castShadow = true;
    const light = new THREE.DirectionalLight(0xffffff, 3);
    light.position.set(4, 0, 4);
    light.castShadow = true;
    const options = { environmentIntensity: 0, shadows: true, ambientOcclusion: false, shadowSoftness: 0 };
    const image = await render([floor, glass, light], options, 64, 64);
    expect(image.pixel(13, 32)[0]).toBeLessThan(10);
    expect(image.pixel(50, 32)[0]).toBeGreaterThan(100);
  });
  it("leaves tone mapping and the output color space of custom shaders to the final image", async() => {
    const material = new THREE.ShaderMaterial({
      vertexShader: "void main(){gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0);}",
      fragmentShader: `void main(){
  gl_FragColor=vec4(1.0,0.0,0.0,1.0);
  #include <tonemapping_fragment>
  #include <colorspace_fragment>
}`,
    });

    expect((await render([new THREE.Mesh(triangle(0).geometry, material)])).pixel()).toEqual([255, 0, 0, 255]);
  });
  it("blends the envMap of Basic, Lambert and Phong through combine and reflectivity", async() => {
    const envMap = new THREE.DataTexture(new Uint8Array([0, 255, 0, 255]), 1, 1);
    envMap.mapping = THREE.EquirectangularReflectionMapping;
    envMap.needsUpdate = true;

    const pixel = async(parameters: THREE.MeshBasicMaterialParameters) => (await render(
      [new THREE.Mesh(triangle(0).geometry, new THREE.MeshBasicMaterial({ color: 0xffffff, envMap, ...parameters }))],
    )).pixel();

    expect(await pixel({})).toEqual([0, 255, 0, 255]);
    expect(await pixel({ reflectivity: 0 })).toEqual([255, 255, 255, 255]);
    expect(await pixel({ color: 0xff0000, combine: THREE.MixOperation })).toEqual([0, 255, 0, 255]);
    expect(await pixel({ color: 0xff0000, combine: THREE.AddOperation })).toEqual([255, 255, 0, 255]);
  });
  it("orients equirectangular and cube envMaps like Three", async() => {
    // A mirror tilted 45 degrees reflects the camera ray straight up.
    const mirror = (envMap: THREE.Texture) => {
      const mesh = new THREE.Mesh(new THREE.PlaneGeometry(4, 4),
        new THREE.MeshBasicMaterial({ color: 0xffffff, envMap }));

      mesh.position.z = -2;
      mesh.rotation.x = -Math.PI / 4;

      return mesh;
    };

    // Data rows run bottom-up: the second row is the sky.
    const equirect = new THREE.DataTexture(new Uint8Array([255, 0, 0, 255, 0, 255, 0, 255]), 1, 2);
    equirect.mapping = THREE.EquirectangularReflectionMapping;
    equirect.magFilter = THREE.NearestFilter;
    equirect.needsUpdate = true;
    expect((await render([mirror(equirect)])).pixel()).toEqual([0, 255, 0, 255]);

    const face = (r: number, g: number) => {
      const texture = new THREE.DataTexture(new Uint8Array([r, g, 0, 255]), 1, 1);
      texture.needsUpdate = true;

      return texture;
    };

    // Only +y is green.
    const red = () => face(255, 0);
    const cube = new THREE.CubeTexture([red(), red(), face(0, 255), red(), red(), red()]);
    cube.needsUpdate = true;
    expect((await render([mirror(cube)])).pixel()).toEqual([0, 255, 0, 255]);
  });
  it("pushes depth back by polygonOffsetFactor times the triangle's slope", async() => {
    const tilted = (color: number, factor: number) => {
      const mesh = new THREE.Mesh(new THREE.PlaneGeometry(4, 4), new THREE.MeshBasicMaterial({
        color, polygonOffset: factor !== 0, polygonOffsetFactor: factor }));

      mesh.position.z = -2;
      mesh.rotation.x = -1;

      return mesh;
    };

    expect((await render([tilted(0x00ff00, 0), tilted(0xff0000, 4)])).pixel()).toEqual([0, 255, 0, 255]);
    expect((await render([tilted(0x00ff00, 4), tilted(0xff0000, 0)])).pixel()).toEqual([255, 0, 0, 255]);
  });
  it("applies scene.fog, linear or exponential, to the materials that accept it", async() => {
    const linear = await render([triangle(0xff0000)], {}, 32, 32, scene => {
      scene.fog = new THREE.Fog(0x0000ff, 0, 1);
    });

    expect(linear.pixel()).toEqual([0, 0, 255, 255]);
    expect(linear.pixel(0, 0)).toEqual([0, 0, 0, 0]);

    const dense = await render([triangle(0xff0000)], {}, 32, 32, scene => {
      scene.fog = new THREE.FogExp2(0x0000ff, .5);
    });

    // 1 - exp(-(0.5 * 2)^2) = 0.632 of fog at depth 2.
    expect(dense.pixel()[2]).toBeGreaterThan(dense.pixel()[0]);
    expect(dense.pixel()[0]).toBeGreaterThan(100);
    const unfogged = triangle(0xff0000);
    unfogged.material.fog = false;

    const skipped = await render([unfogged], {}, 32, 32, scene => {
      scene.fog = new THREE.Fog(0x0000ff, 0, 1);
    });

    expect(skipped.pixel()).toEqual([255, 0, 0, 255]);
  });
  it("reads the toon ramp over dot(n, l) from -1 to 1, lighting the side that faces away", async() => {
    const ramp = new THREE.DataTexture(new Uint8Array([64, 64, 64, 255, 255, 255, 255, 255]), 2, 1);
    ramp.needsUpdate = true;
    const material = new THREE.MeshToonMaterial({ color: 0xffffff, gradientMap: ramp });
    const toon = new THREE.Mesh(triangle(0).geometry, material);

    const lit = async(y: number, z: number) => {
      const light = new THREE.DirectionalLight(0xffffff, Math.PI);
      light.position.set(0, y, z);

      return (await render([toon, light], { environmentIntensity: 0, shadows: false })).pixel()[0];
    };

    // The triangle faces +z. From behind (dot = -0.5), the dark texel still lights it; from the front, the bright one.
    const behind = await lit(1, -.577), front = await lit(0, 1);
    expect(behind).toBeGreaterThan(60);
    expect(behind).toBeLessThan(160);
    expect(front).toBe(255);
  });
  it("tints a spot light with its projected map", async() => {
    const wall = () => {
      const mesh = new THREE.Mesh(new THREE.PlaneGeometry(6, 6), new THREE.MeshLambertMaterial({ color: 0xffffff }));
      mesh.position.z = -2;

      return mesh;
    };

    const spot = (map: THREE.Texture | null) => {
      const light = new THREE.SpotLight(0xffffff, 30, 0, Math.PI / 8, .2, 0);
      light.position.set(0, 0, 1);
      light.map = map;

      return light;
    };

    const green = new THREE.DataTexture(new Uint8Array([0, 255, 0, 255]), 1, 1);
    green.needsUpdate = true;
    const options = { environmentIntensity: 0, shadows: false };
    expect((await render([wall(), spot(null)], options)).pixel()[0]).toBeGreaterThan(100);
    const tinted = (await render([wall(), spot(green)], options)).pixel();
    expect(tinted[0]).toBe(0);
    expect(tinted[1]).toBeGreaterThan(100);
  });
  it("does not multiply a premultiplied custom shader by its alpha again", async() => {
    const material = new THREE.ShaderMaterial({
      transparent: true,
      premultipliedAlpha: true,
      vertexShader: "void main(){gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0);}",
      fragmentShader: "void main(){gl_FragColor=vec4(0.5,0.0,0.0,0.5);}",
    });

    const result = await render([triangle(0xffffff, -3), new THREE.Mesh(triangle(0).geometry, material)]);
    // ONE, ONE_MINUS_SRC_ALPHA: 0.5 + 1 * 0.5 in red, 0 + 1 * 0.5 elsewhere.
    expect(result.pixel()[0]).toBe(255);
    expect(result.pixel()[1]).toBeGreaterThan(180);
    expect(result.pixel()[1]).toBeLessThan(195);
  });
  it("keeps chunk globals computed before <opaque_fragment> visible after it", async() => {
    const mesh = triangle(0xffffff);

    mesh.material.onBeforeCompile = shader => {
      shader.fragmentShader = shader.fragmentShader
        .replace("#include <common>", "#include <common>\nvec3 cpuStage;")
        .replace("#include <color_fragment>", "#include <color_fragment>\ncpuStage = vec3(0.0, 1.0, 0.0);")
        .replace("#include <dithering_fragment>", "#include <dithering_fragment>\ngl_FragColor.rgb = cpuStage;");
    };

    expect((await render([mesh])).pixel()).toEqual([0, 255, 0, 255]);
  });
  it("tone maps the final linear accumulation once", async() => {
    const mesh = triangle(0xffffff);
    mesh.material.transparent = true;
    mesh.material.opacity = .5;
    const result = await render([mesh, triangle(0x000000, -3)], { tonemapping: "aces", exposure: 1 });
    // Half white over black in linear light becomes 0.5 before ACES, not the average of two already mapped values.
    const expected = encodeColor([.5, .5, .5], 1, "aces");
    expect(result.pixel().slice(0, 3)).toEqual(expected);
  });
  it("prefilters the environment by roughness", async() => {
    const env = await CpuEnvironment.load({ kind: "gradient", topColor: "#ffffff", bottomColor: "#000000", exponent: 1 });
    const up = [0, 1, 0];
    const sharp = env.sample(up, 0, 0)[0], rough = env.sample(up, 0, 1)[0], mid = env.sample(up, 0, .5)[0];
    expect(sharp).toBeCloseTo(1, 5);
    expect(rough).toBeLessThan(sharp);
    expect(rough).toBeGreaterThan(.5);
    expect(mid).toBeLessThan(sharp);
    expect(mid).toBeGreaterThan(rough);
    expect(env.sample([1, 0, 0], 0, 0, true)[0]).toBeCloseTo(.5, 1);
  });
});

describe("effects", () => {
  it("darkens ambient light under nearby geometry with ray traced occlusion", async() => {
    // White floor with a lid just in front of its center; ambient light only.
    const floor = new THREE.Mesh(new THREE.PlaneGeometry(6, 6), new THREE.MeshLambertMaterial({ color: 0xffffff }));
    floor.position.z = -3;
    floor.receiveShadow = true;

    const lid = new THREE.Mesh(new THREE.PlaneGeometry(1.5, 1.5),
      new THREE.MeshLambertMaterial({ color: 0xffffff, side: THREE.DoubleSide }));

    lid.position.z = -2.9;
    lid.castShadow = true;
    const light = new THREE.AmbientLight(0xffffff, Math.PI);
    const off = await render([floor, lid, light], { environmentIntensity: 0, ambientOcclusion: false });
    const on = await render([floor, lid, light], { environmentIntensity: 0, ambientOcclusion: true, aoDistance: 1 });
    // Floor right at the lid edge loses hemisphere; the far corner and the lid itself do not change.
    expect(on.pixel(8, 16)[0]).toBeLessThan(off.pixel(8, 16)[0]);
    expect(on.pixel(1, 1)).toEqual(off.pixel(1, 1));
    expect(on.pixel(16, 16)).toEqual(off.pixel(16, 16));
  });
  it("softens shadow edges with an angular light size", async() => {
    const floor = new THREE.Mesh(new THREE.PlaneGeometry(6, 6), new THREE.MeshLambertMaterial({ color: 0xffffff }));
    floor.position.z = -3;
    floor.receiveShadow = true;
    const blocker = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), new THREE.MeshLambertMaterial({ color: 0xffffff }));
    blocker.position.z = -2;
    blocker.castShadow = true;
    // Tilted light: the shadow falls beside the blocker, where the camera can see it.
    const light = new THREE.DirectionalLight(0xffffff, 3);
    light.position.set(4, 0, 4);
    light.castShadow = true;
    const options = { environmentIntensity: 0, shadows: true, ambientOcclusion: false };
    const distinct = (image: Uint8ClampedArray) => new Set(Array.from(image).filter((_, i) => i % 4 === 0)).size;
    const hard = await render([floor, blocker, light], { ...options, shadowSoftness: 0 }, 64, 64);
    const soft = await render([floor, blocker, light], { ...options, shadowSoftness: 10 }, 64, 64);
    expect(distinct(hard.image)).toBeLessThanOrEqual(2);
    expect(distinct(soft.image)).toBeGreaterThanOrEqual(3);
  });
  it("fades distant surfaces into the fog color without touching the background", async() => {
    const fog = { fogEnabled: true, fogColor: "#0000ff", fogNear: 0, fogFar: 1 };
    const fogged = await render([triangle(0xff0000)], fog);
    expect(fogged.pixel()).toEqual([0, 0, 255, 255]);
    expect(fogged.pixel(0, 0)).toEqual([0, 0, 0, 0]);
    expect((await render([triangle(0xff0000)], { ...fog, fogEnabled: false })).pixel()).toEqual([255, 0, 0, 255]);
  });
  it("applies vignette and deterministic grain at resolve", async() => {
    const wall = new THREE.Mesh(new THREE.PlaneGeometry(20, 20), new THREE.MeshBasicMaterial({ color: 0xffffff }));
    wall.position.z = -2;
    const plain = await render([wall]);
    const vignetted = await render([wall], { vignette: 1 });
    expect(vignetted.pixel(16, 16)).toEqual(plain.pixel(16, 16));
    expect(vignetted.pixel(0, 0)[0]).toBeLessThan(plain.pixel(0, 0)[0]);
    const grainy = await render([wall], { grain: .5 }), again = await render([wall], { grain: .5 });
    expect(grainy.image).toEqual(again.image);
    expect(grainy.image).not.toEqual(plain.image);
  });
});
