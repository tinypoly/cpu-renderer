import { afterEach, describe, expect, it, vi } from "vitest";
import * as THREE from "three";
import { serializeScene, serializeTexture } from "./sceneSerialization.js";
import { sampleTexture, textureFootprint, textureLod, srgbToLinear } from "./texture.js";
import { CpuShader } from "./glsl.js";
import { CpuRasterizer } from "./rasterizer.js";
import { CpuEnvironment } from "./environment.js";
import { serializeCamera } from "./camera.js";
import { DEFAULT_FRAME_SETTINGS, type FrameSettings } from "./frameSettings.js";
import { sceneSignature } from "./sceneSignature.js";
import { CpuMaterial } from "./material.js";

afterEach(() => vi.unstubAllGlobals());

function checker(size = 64) {
  const data = new Uint8Array(size * size * 4);

  for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
    const i = (y * size + x) * 4, color = (x + y) % 2 ? 255 : 0;
    data.set([color, color, color, 255], i);
  }

  const texture = new THREE.DataTexture(data, size, size);
  texture.generateMipmaps = true;
  texture.minFilter = THREE.LinearMipmapLinearFilter;
  texture.magFilter = THREE.NearestFilter;

  return texture;
}

function solid(width: number, height: number, color: number[]) {
  const data = new Uint8Array(width * height * 4);
  for (let i = 0; i < data.length; i += 4) data.set(color, i);

  return { width, height, data };
}

async function levels() {
  const texture = new THREE.DataTexture(new Uint8Array(4), 1, 1);
  texture.mipmaps = [solid(4, 4, [255, 0, 0, 255]), solid(2, 2, [0, 255, 0, 255]), solid(1, 1, [0, 0, 255, 255])];
  texture.minFilter = THREE.LinearMipmapLinearFilter;

  return serializeTexture(texture);
}

describe("mipmap transport and generation", () => {
  it("generates the whole chain and averages sRGB in linear light", async() => {
    const texture = checker(4);
    const linear = await serializeTexture(texture);
    expect(linear.mipmaps?.map(m => [m.width, m.height])).toEqual([[2, 2], [1, 1]]);
    expect(Array.from(linear.mipmaps![1].data)).toEqual([128, 128, 128, 255]);
    texture.colorSpace = THREE.SRGBColorSpace;
    const srgb = await serializeTexture(texture);
    expect(Array.from(srgb.mipmaps![1].data)).toEqual([188, 188, 188, 255]);
    expect(sampleTexture(srgb, [.5, .5], false, { lod: 2 })[0]).toBeCloseTo(srgbToLinear(188 / 255), 6);
    texture.generateMipmaps = false;
    expect((await serializeTexture(texture)).mipmaps).toBeUndefined();
  });

  it("includes odd-dimension edges and preserves HDR and alpha", async() => {
    const data = new Float32Array(3 * 5 * 4);
    for (let i = 0; i < 15; i++) data.set([i === 14 ? 30 : 0, 4, 8, .25], i * 4);
    const texture = new THREE.DataTexture(data, 3, 5, THREE.RGBAFormat, THREE.FloatType);
    texture.generateMipmaps = true;
    const snapshot = await serializeTexture(texture);
    expect(snapshot.mipmaps?.map(m => [m.width, m.height])).toEqual([[1, 2], [1, 1]]);
    expect(Array.from(snapshot.mipmaps![1].data)).toEqual([2, 4, 8, .25]);
  });

  it("uses authored level zero and validates the dimensions of subsequent levels", async() => {
    const snapshot = await levels();
    expect(snapshot.width).toBe(4);
    expect(Array.from(snapshot.data.slice(0, 4))).toEqual([255, 0, 0, 255]);
    expect(Array.from(snapshot.mipmaps![0].data.slice(0, 4))).toEqual([0, 255, 0, 255]);
    const texture = checker(4);
    texture.mipmaps = [solid(4, 4, [0, 0, 0, 255]), solid(3, 3, [0, 0, 0, 255])];
    await expect(serializeTexture(texture)).rejects.toThrow("mipmap dimensions");
    await expect(serializeTexture(new THREE.CompressedTexture([], 1, 1))).rejects.toThrow("compressed textures");
  });

  it.each([false, true])("shares all generated levels across sampler clones (shared=%s)", async shared => {
    vi.stubGlobal("crossOriginIsolated", shared);
    const map = checker(8), other = map.clone();
    other.repeat.set(2, 3);
    other.minFilter = THREE.NearestMipmapNearestFilter;
    const scene = new THREE.Scene(), geometry = new THREE.PlaneGeometry();
    scene.add(new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({ map })),
      new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({ map: other })));
    const { scene: snapshot, transfer } = await serializeScene(scene);
    const [a, b] = snapshot.textures;
    expect(a.mipmaps).toBe(b.mipmaps);
    expect(a.minFilter).not.toBe(b.minFilter);

    for (const level of [a, ...a.mipmaps!]) {
      expect(level.data.buffer instanceof SharedArrayBuffer).toBe(shared);
      expect(transfer.filter(buffer => buffer === level.data.buffer)).toHaveLength(shared ? 0 : 1);
    }

    const cloned = structuredClone(snapshot, { transfer });
    expect(cloned.textures[0].mipmaps).toHaveLength(3);
    expect(map.image.data!.byteLength).toBe(256);
  });

  it("detects filter/generation changes and does not reuse incompatible clone chains", async() => {
    const map = checker(4), other = map.clone();
    other.generateMipmaps = false;
    const scene = new THREE.Scene(), geometry = new THREE.PlaneGeometry();
    scene.add(new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({ map })),
      new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({ map: other })));
    const before = sceneSignature(scene);
    map.minFilter = THREE.NearestMipmapNearestFilter;
    expect(sceneSignature(scene)).not.toBe(before);
    const result = (await serializeScene(scene)).scene.textures;
    expect(result[0].mipmaps).toHaveLength(2);
    expect(result[1].mipmaps).toBeUndefined();
  });

  it("reads compressed textures transcoded to RGBA and rejects block formats", async() => {
    const mipmaps = [solid(4, 4, [255, 0, 0, 255]), solid(2, 2, [0, 255, 0, 255]), solid(1, 1, [0, 0, 255, 255])];

    // Three's types only list block formats, but KTX2Loader's RGBA32 fallback builds exactly this at runtime.
    const rgba = new THREE.CompressedTexture(mipmaps, 4, 4, THREE.RGBAFormat as unknown as THREE.CompressedPixelFormat,
      THREE.UnsignedByteType);

    rgba.minFilter = THREE.LinearMipmapLinearFilter;
    const snapshot = await serializeTexture(rgba);
    expect([snapshot.width, snapshot.height, snapshot.flipY]).toEqual([4, 4, false]);
    expect(snapshot.mipmaps?.map(m => [m.width, m.height])).toEqual([[2, 2], [1, 1]]);
    expect(sampleTexture(snapshot, [.5, .5], false, { lod: 2 })).toEqual([0, 0, 1, 1]);

    const dxt = new THREE.CompressedTexture(mipmaps, 4, 4, THREE.RGBA_S3TC_DXT5_Format);
    await expect(serializeTexture(dxt)).rejects.toThrow(/compressed textures/);
  });

  it("preserves authored cube levels in the equirectangular proxy", async() => {
    const faces = (size: number, color: number[]) => Array.from({ length: 6 }, () => {
      const pixels = solid(size, size, color);

      return new THREE.DataTexture(pixels.data, size, size);
    });

    const cube = new THREE.CubeTexture(faces(2, [255, 0, 0, 255]));
    cube.mipmaps = [new THREE.CubeTexture(faces(1, [0, 255, 0, 255]))];
    const snapshot = await serializeTexture(cube);
    expect(snapshot.mipmaps?.map(m => [m.width, m.height])).toEqual([[4, 2], [2, 1], [1, 1]]);
    expect(sampleTexture(snapshot, [.5, .5], false, { lod: 1 })).toEqual([0, 1, 0, 1]);
  });
});

describe("mipmap filtering", () => {
  it.each([
    [THREE.NearestFilter, [1, 0, 0, 1]],
    [THREE.LinearFilter, [1, 0, 0, 1]],
    [THREE.NearestMipmapNearestFilter, [0, 1, 0, 1]],
    [THREE.LinearMipmapNearestFilter, [0, 1, 0, 1]],
    [THREE.NearestMipmapLinearFilter, [0, .75, .25, 1]],
    [THREE.LinearMipmapLinearFilter, [0, .75, .25, 1]],
  ])("implements minification filter %s", async(minFilter, expected) => {
    const texture = { ...await levels(), minFilter: minFilter as number };
    expect(sampleTexture(texture, [.5, .5], false, { lod: 1.25 })).toEqual(expected);
    expect(sampleTexture(texture, [.5, .5], false, { lod: -10 })).toEqual([1, 0, 0, 1]);
  });

  it("uses independent min/mag filters and clamps to available levels", async() => {
    const texture = await serializeTexture(checker(4));
    texture.minFilter = THREE.LinearFilter;
    expect(sampleTexture(texture, [.5, .5], false, { lod: -1 })[0]).toBe(0);
    expect(sampleTexture(texture, [.5, .5], false, { lod: 1 })[0]).toBe(.5);
    const mip = await levels();
    expect(sampleTexture(mip, [.5, .5], false, { lod: 100 })).toEqual([0, 0, 1, 1]);
    expect(sampleTexture(mip, [.5, .5], false, { lod: Infinity })).toEqual([0, 0, 1, 1]);
  });

  it.each([
    [THREE.NearestMipmapNearestFilter, 0], [THREE.NearestMipmapLinearFilter, 0],
    [THREE.LinearMipmapNearestFilter, .5], [THREE.LinearMipmapLinearFilter, .5],
  ])("uses the spatial filter within mip level one (%s)", async(minFilter, expected) => {
    const texture = await levels();
    texture.minFilter = minFilter;
    texture.mipmaps![0].data = checker(2).image.data as Uint8Array;
    expect(sampleTexture(texture, [.5, .5], false, { lod: 1 })[0]).toBe(expected);
  });

  it("derives LOD from unwrapped, transformed UV gradients and bias", async() => {
    const texture = await levels();
    texture.matrix = [2, 0, 0, 0, 4, 0, 100, 0, 1];
    const gradients = { dx: [.125, 0], dy: [0, .125] };
    expect(textureLod(texture, gradients)).toBe(1);
    expect(textureLod(texture, gradients, false)).toBe(-1);
    expect(sampleTexture(texture, [.99, .99], true, gradients)).toEqual([0, 1, 0, 1]);
    expect(sampleTexture(texture, [.99, .99], true, { ...gradients, bias: 1 })).toEqual([0, 0, 1, 1]);
  });
});

describe("anisotropic filtering", () => {
  // A footprint four texels wide and one texel tall on the 4x4 base level.
  const stretched = { dx: [1, 0], dy: [0, .25] };

  it("selects the level from the minor axis, capped by the texture's anisotropy", async() => {
    const texture = await levels();
    expect(sampleTexture(texture, [.5, .5], false, stretched)).toEqual([0, 0, 1, 1]);
    texture.anisotropy = 2;
    expect(textureFootprint(texture, stretched)).toEqual({ lod: 1, taps: 2, axis: [1, 0] });
    expect(sampleTexture(texture, [.5, .5], false, stretched)).toEqual([0, 1, 0, 1]);
    texture.anisotropy = 16;
    expect(textureFootprint(texture, stretched)).toEqual({ lod: 0, taps: 4, axis: [1, 0] });
    expect(sampleTexture(texture, [.5, .5], false, stretched)).toEqual([1, 0, 0, 1]);
    // Explicit LOD and magnification stay isotropic.
    expect(textureFootprint(texture, { ...stretched, lod: 1 }).taps).toBe(1);
    expect(textureFootprint(texture, { dx: [.1, 0], dy: [0, .05] }).taps).toBe(1);
  });

  it("averages taps along the major axis, through the texture transform", async() => {
    const halves = new Uint8Array(4 * 4 * 4);
    for (let i = 0; i < 16; i++) halves.set(i % 4 < 2 ? [255, 0, 0, 255] : [0, 255, 0, 255], i * 4);
    const texture = new THREE.DataTexture(halves, 4, 4);
    texture.generateMipmaps = true;
    texture.minFilter = THREE.NearestMipmapNearestFilter;
    texture.magFilter = THREE.NearestFilter;
    texture.wrapS = THREE.RepeatWrapping;
    texture.anisotropy = 16;
    texture.rotation = Math.PI / 2;
    texture.updateMatrix();
    const snapshot = await serializeTexture(texture);
    expect(snapshot.anisotropy).toBe(16);
    // After the quarter turn, the screen-space y gradient runs along the texture's u axis.
    const result = sampleTexture(snapshot, [.5, .5], true, { dx: [.25, 0], dy: [0, 1] });
    expect(result.map(c => Math.round(c * 1e6) / 1e6)).toEqual([.5, .5, 0, 1]);
    expect(sampleTexture(snapshot, [.5, .5], true, { dx: [.25, 0], dy: [0, 1], lod: 0 })).not.toEqual(result);
  });

  it("changes the scene signature", () => {
    const map = checker(4), scene = new THREE.Scene();
    scene.add(new THREE.Mesh(new THREE.PlaneGeometry(), new THREE.MeshBasicMaterial({ map })));
    const before = sceneSignature(scene);
    map.anisotropy = 8;
    expect(sceneSignature(scene)).not.toBe(before);
  });
});

describe("shader mipmap sampling", () => {
  it("supports explicit LOD, gradients and extension aliases without a quad", async() => {
    const texture = await levels();

    const context = { texture: (_: unknown, uv: number[], sampling?: Parameters<typeof sampleTexture>[3]) =>
      sampleTexture(texture, uv, false, sampling) };

    for (const expression of ["textureLod(map,vec2(0.5),1.0)", "texture2DLodEXT(map,vec2(0.5),1.0)",
      "textureGrad(map,vec2(0.5),vec2(0.5,0.0),vec2(0.0,0.5))",
      "texture2DGradEXT(map,vec2(0.5),vec2(0.5,0.0),vec2(0.0,0.5))"])
      expect(new CpuShader(`uniform sampler2D map;void main(){gl_FragColor=${expression};}`)
        .run({ map: "map" }, context)!.gl_FragColor).toEqual([0, 1, 0, 1]);
  });

  it("derives implicit LOD after shader UV arithmetic and projection", async() => {
    const texture = await levels();

    const context = { texture: (_: unknown, uv: number[], sampling?: Parameters<typeof sampleTexture>[3]) =>
      sampleTexture(texture, uv, false, sampling) };

    const inputs = [0, 1, 2, 3].map(i => ({ map: "map", uv: [(i & 1) * .125, (i >> 1) * .125] }));
    for (const expression of ["texture2D(map,uv*4.0)", "texture(map,uv,2.0)", "texture2DProj(map,vec3(uv,0.25))"])
      for (const result of new CpuShader(`uniform sampler2D map;varying vec2 uv;void main(){gl_FragColor=${expression};}`)
        .runQuad(inputs, context)) expect(result!.gl_FragColor).toEqual([0, 1, 0, 1]);
  });
});

function render(renderer: CpuRasterizer) {
  for (const _ of renderer.prepare()) { /* drain */ }

  const pixels = new Uint8ClampedArray(16 * 16 * 4);

  for (const bucket of renderer.buckets) {
    const job = renderer.renderBucket(bucket);
    let step = job.next();
    while (!step.done) step = job.next();
    for (let y = 0; y < bucket.height; y++) pixels.set(
      step.value.pixels.subarray(y * bucket.width * 4, (y + 1) * bucket.width * 4),
      ((bucket.y + y) * 16 + bucket.x) * 4);
  }

  return pixels;
}

async function renderScene(material: THREE.Material, geometry = new THREE.PlaneGeometry(2, 2),
  options: Partial<FrameSettings> = {}) {
  const scene = new THREE.Scene(), mesh = new THREE.Mesh(geometry, material);
  mesh.position.z = -2;
  scene.add(mesh);
  const camera = serializeCamera(new THREE.OrthographicCamera(-1, 1, 1, -1, .1, 10));

  const renderer = new CpuRasterizer((await serializeScene(scene)).scene, camera,
    { ...DEFAULT_FRAME_SETTINGS, tileSize: 9, maxSamples: 1, shadows: false, ambientOcclusion: false,
      tonemapping: "linear", backgroundMode: "transparent", ...options }, 16, 16, new CpuEnvironment());

  return { renderer, pixels: render(renderer) };
}

describe("raster mipmap selection", () => {
  it.each(["native", "shader"])("minifies a checker without aliasing (%s)", async kind => {
    const map = checker();

    const material = kind === "native" ? new THREE.MeshBasicMaterial({ map }) : new THREE.ShaderMaterial({
      uniforms: { map: { value: map } },
      vertexShader: "varying vec2 vUv;void main(){vUv=uv;gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0);}",
      fragmentShader: "uniform sampler2D map;varying vec2 vUv;void main(){gl_FragColor=texture2D(map,vUv);}",
    });

    const { renderer, pixels } = await renderScene(material);
    for (let i = 0; i < pixels.length; i += 4) expect(Array.from(pixels.slice(i, i + 4))).toEqual([188, 188, 188, 255]);

    const adopted = new CpuRasterizer(renderer.scene, renderer.camera, renderer.settings, 16, 16,
      new CpuEnvironment(), {}, renderer.preparedGeometry);

    expect(render(adopted)).toEqual(pixels);
  });

  it.each([false, true])("selects authored levels from transformed coordinates (shader=%s)", async shader => {
    const texture = checker(4);
    texture.mipmaps = [solid(4, 4, [255, 0, 0, 255]), solid(2, 2, [0, 255, 0, 255]), solid(1, 1, [0, 0, 255, 255])];
    texture.repeat.set(8, 8);

    const material = shader ? new THREE.ShaderMaterial({
      uniforms: { map: { value: texture } },
      vertexShader: "varying vec2 vUv;void main(){vUv=uv;gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0);}",
      fragmentShader: "uniform sampler2D map;varying vec2 vUv;void main(){gl_FragColor=texture2D(map,vUv*8.0);}",
    }) : new THREE.MeshBasicMaterial({ map: texture });

    const { pixels } = await renderScene(material);
    for (let i = 0; i < pixels.length; i += 4) expect(Array.from(pixels.slice(i, i + 4))).toEqual([0, 255, 0, 255]);
  });

  it("honors UV channels, repeats and alpha maps in native materials", async() => {
    const map = checker(16);
    map.channel = 1;
    map.repeat.set(4, 4);
    map.wrapS = map.wrapT = THREE.RepeatWrapping;
    const geometry = new THREE.PlaneGeometry(2, 2);
    geometry.setAttribute("uv1", geometry.getAttribute("uv").clone());
    geometry.getAttribute("uv").array.fill(0);

    const { pixels } = await renderScene(new THREE.MeshBasicMaterial({ map, alphaMap: map, transparent: true }),
      geometry);

    const center = (8 * 16 + 8) * 4;
    expect(Array.from(pixels.slice(center, center + 4))).toEqual([188, 188, 188, 128]);
  });

  it("alpha tests cutouts at the shading footprint in the irradiance cache passes", async() => {
    // Alternating 0/255 alpha texels: the base level fails the test at half the pixel centers, the minified level
    // at none. Cache placement and camera-ray lookups must agree with the fragment pass.
    const alphaMap = checker();
    const alpha = vi.spyOn(CpuMaterial.prototype, "alpha");

    const { pixels } = await renderScene(new THREE.MeshLambertMaterial({ alphaMap, alphaTest: .4 }), undefined,
      { globalIllumination: true, giCache: true, giCacheSpacing: 4, giSamples: 1 });

    expect(alpha).toHaveBeenCalled();
    for (const [, gradients] of alpha.mock.calls) expect(gradients?.uv?.dx).toBeDefined();
    for (let i = 3; i < pixels.length; i += 4) expect(pixels[i]).toBe(255);
    alpha.mockRestore();
  });
});
