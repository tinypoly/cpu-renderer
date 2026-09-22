import { describe, expect, it } from "vitest";
import * as THREE from "three";
import { linearToSrgb } from "./texture.js";
import { CpuShader } from "./glsl.js";
import { CpuRasterizer } from "./rasterizer.js";
import { CpuEnvironment } from "./environment.js";
import { serializeScene } from "./sceneSerialization.js";
import { serializeCamera } from "./camera.js";
import { DEFAULT_FRAME_SETTINGS } from "./frameSettings.js";

const context = { texture: (_sampler: unknown, uv: number[]) => [uv[0] ** 2, uv[1], 0, 1] };
const inputs = [0, 1, 2, 3].map(i => ({ uv: [2 + (i & 1), 3 + (i >> 1)] }));

describe("GLSL fragment derivatives", () => {
  it("differences intermediate scalar and vector expressions in GL quad order", () => {
    const shader = new CpuShader(`varying vec2 uv;
void main() { vec2 v=uv*uv; gl_FragColor=vec4(dFdx(v),dFdy(v)); }`);

    expect(shader.usesDerivatives).toBe(true);
    for (const result of shader.runQuad(inputs, context))
      expect(result!.gl_FragColor).toEqual([5, 0, 0, 7]);

    const width = new CpuShader(`varying vec2 uv;
void main() { float v=sin(uv.x)+uv.y*uv.y; gl_FragColor=vec4(fwidth(v)); }`);

    for (const result of width.runQuad(inputs, context))
      expect((result!.gl_FragColor as number[])[0]).toBeCloseTo(Math.abs(Math.sin(3) - Math.sin(2)) + 7);
  });

  it("handles functions, inout, uniform loops and dependent derivative calls", () => {
    const shader = new CpuShader(`varying vec2 uv;
float increment(inout float v) { v+=1.0; return v*v; }
void main() {
  float v=uv.x; float sum=0.0;
  for(int i=0;i<3;i++) { sum+=dFdx(increment(v)); }
  float first=dFdx(uv.x*uv.x);
  gl_FragColor=vec4(sum,v,dFdy(first*uv.y),1.0);
}`);

    const result = shader.runQuad(inputs, context);
    expect(result[0]!.gl_FragColor).toEqual([27, 5, 5, 1]);
    expect(result[1]!.gl_FragColor).toEqual([27, 6, 5, 1]);
    expect(inputs[0].uv).toEqual([2, 3]);
  });

  it("isolates mutable input vectors across helper replays", () => {
    const shader = new CpuShader(`varying vec2 uv;
float modify(inout vec2 p) { p.x+=1.0; return p.x*p.x; }
void main(){ float d=dFdx(modify(uv)); gl_FragColor=vec4(d,uv.x,0.0,1.0); }`);

    expect(shader.runQuad(inputs, context)[0]!.gl_FragColor).toEqual([7, 3, 0, 1]);
    expect(inputs[0].uv).toEqual([2, 3]);
  });

  it("evaluates texture results at the helper inputs", () => {
    const shader = new CpuShader(`varying vec2 uv; uniform sampler2D map;
void main() { vec4 sampled=texture2D(map,uv); gl_FragColor=vec4(dFdx(sampled).r,dFdy(sampled).g,0.0,1.0); }`);

    expect(shader.runQuad(inputs.map(i => ({ ...i, map: "map" })), context)[0]!.gl_FragColor).toEqual([5, 1, 0, 1]);
  });

  it("preserves derivatives before discard and reports divergent rendezvous", () => {
    const after = new CpuShader(`varying vec2 uv;
void main() { float d=dFdx(uv.x); if(uv.x>2.5) discard; gl_FragColor=vec4(d); }`);

    const results = after.runQuad(inputs, context);
    expect(results[0]!.gl_FragColor).toEqual([1, 1, 1, 1]);
    expect(results[1]).toBeNull();

    const before = new CpuShader(`varying vec2 uv;
void main() { if(uv.x>2.5) discard; gl_FragColor=vec4(dFdx(uv.x)); }`);

    expect(() => before.runQuad(inputs, context)).toThrow("non-uniform");

    const branch = new CpuShader(`varying vec2 uv;
void main() { if(uv.x>2.5) gl_FragColor=vec4(dFdx(uv.x)); else gl_FragColor=vec4(dFdy(uv.y)); }`);

    expect(() => branch.runQuad(inputs, context)).toThrow("non-uniform");
  });

  it("keeps scalar execution, validates operands and bounds helper work", () => {
    expect(new CpuShader("void main(){gl_FragColor=vec4(1.0);}").usesDerivatives).toBe(false);
    expect(() => new CpuShader("void main(){gl_FragColor=vec4(dFdx(1));}")).toThrow("float");
    expect(() => new CpuShader("void main(){gl_FragColor=vec4(dFdy());}")).toThrow("expects one");

    const shader = new CpuShader(`varying vec2 uv;
void main(){ float v=0.0; for(int i=0;i<100;i++) v+=dFdx(uv.x); gl_FragColor=vec4(v); }`);

    expect(() => shader.runQuad(inputs, { ...context, budget: 100 })).toThrow("budget exceeded");
    expect(() => shader.run(inputs[0], context)).toThrow("fragment quad");
  });
});

const vertexShader = `varying vec2 vUv;
void main(){ vUv=uv; gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0); }`;

const settings = { ...DEFAULT_FRAME_SETTINGS, maxSamples: 1, tileSize: 16,
  shadows: false, ambientOcclusion: false, backgroundMode: "transparent" as const, tonemapping: "linear" as const };

function image(renderer: CpuRasterizer) {
  for (const _ of renderer.prepare()) { /* drain */ }

  const result = new Uint8ClampedArray(renderer.width * renderer.height * 4);

  for (const bucket of renderer.buckets) {
    const job = renderer.renderBucket(bucket);
    let step = job.next();
    while (!step.done) step = job.next();
    for (let y = 0;y < bucket.height;y++)
      result.set(step.value.pixels.subarray(y * bucket.width * 4,(y + 1) * bucket.width * 4),
        ((bucket.y + y) * renderer.width + bucket.x) * 4);
  }

  return result;
}

async function rendererFor(fragmentShader: string, tileSize = 16, material?: THREE.Material) {
  const scene = new THREE.Scene();

  const mesh = new THREE.Mesh(new THREE.PlaneGeometry(2,2), material ?? new THREE.ShaderMaterial({
    vertexShader, fragmentShader,
  }));

  mesh.position.z = -2;
  scene.add(mesh);
  const snapshot = (await serializeScene(scene)).scene;
  const camera = serializeCamera(new THREE.OrthographicCamera(-1,1,1,-1,.1,10));

  return new CpuRasterizer(snapshot,camera,{ ...settings,tileSize },32,31,new CpuEnvironment());
}

describe("rasterized helper fragments", () => {
  it("matches analytic derivatives on primitive edges, odd resolutions and odd bucket boundaries", async() => {
    const fragment = `varying vec2 vUv;
void main(){gl_FragColor=vec4(dFdx(vUv.x)*32.0,dFdy(vUv.y)*31.0,fwidth(vUv.x)*32.0,1.0);}`;

    const renderer = await rendererFor(fragment,9);
    const rendered = image(renderer);
    expect(rendered.every(v=>v === 255)).toBe(true);
    expect(image(await rendererFor(fragment,16))).toEqual(rendered);

    const adopted = new CpuRasterizer(renderer.scene,renderer.camera,renderer.settings,32,31,
      new CpuEnvironment(),{},renderer.preparedGeometry);

    expect(image(adopted)).toEqual(rendered);
  });

  it("keeps discarded helper fragments from writing pixels", async() => {
    const renderer = await rendererFor(`varying vec2 vUv;
void main(){float d=dFdx(vUv.x)*32.0;if(vUv.x>0.5)discard;gl_FragColor=vec4(d,0.0,0.0,1.0);}`,9);

    const rendered = image(renderer);
    expect(Array.from(rendered.slice((15 * 32 + 15) * 4,(15 * 32 + 15) * 4 + 4))).toEqual([255,0,0,255]);
    expect(Array.from(rendered.slice((15 * 32 + 16) * 4,(15 * 32 + 16) * 4 + 4))).toEqual([0,0,0,0]);
  });

  it.each([.1, 3.5])("matches nonlinear perspective varyings with near clipping at %s", async(near) => {
    const scene = new THREE.Scene();

    const mesh = new THREE.Mesh(new THREE.PlaneGeometry(4, 4), new THREE.ShaderMaterial({
      vertexShader,
      fragmentShader: `varying vec2 vUv;
void main(){ float v=dot(vUv,vUv); gl_FragColor=vec4(abs(dFdx(v))*4.0,abs(dFdy(v))*4.0,0.0,1.0); }`,
    }));

    mesh.position.z = -4;
    mesh.rotation.set(.25, .6, 0);
    scene.add(mesh);
    const snapshot = (await serializeScene(scene)).scene;
    const camera = new THREE.PerspectiveCamera(60, 1, near, 20);

    const renderer = new CpuRasterizer(snapshot, serializeCamera(camera), { ...settings, tileSize: 9 },
      32, 32, new CpuEnvironment());

    const rendered = image(renderer);
    const plane = new THREE.Plane(new THREE.Vector3(0, 0, 1), 0).applyMatrix4(mesh.matrixWorld);
    const ray = new THREE.Raycaster();
    const inverse = mesh.matrixWorld.clone().invert();

    const value = (x: number, y: number) => {
      ray.setFromCamera(new THREE.Vector2(x / 16 - 1, y / 16 - 1), camera);
      const point = ray.ray.intersectPlane(plane, new THREE.Vector3())!.applyMatrix4(inverse);

      return (point.x / 4 + .5) ** 2 + (point.y / 4 + .5) ** 2;
    };

    let checked = 0;

    for (let y = 0; y < 32; y++) for (let x = 0; x < 32; x++) {
      const offset = (y * 32 + x) * 4;
      if (!rendered[offset + 3]) continue;
      const glY = 31 - y, qx = x - (x & 1), qy = glY - (glY & 1);
      const dx = value(qx + 1.5, glY + .5) - value(qx + .5, glY + .5);
      const dy = value(x + .5, qy + 1.5) - value(x + .5, qy + .5);
      const expected = [dx, dy].map(v => Math.round(linearToSrgb(Math.min(1, Math.abs(v) * 4)) * 255));
      expect(rendered[offset]).toBe(expected[0]);
      expect(rendered[offset + 1]).toBe(expected[1]);
      checked++;
    }

    expect(checked).toBeGreaterThan(100);
  });

  it("keeps GL coordinate derivatives correct with supersampling and depth of field", async() => {
    const base = await rendererFor("void main(){gl_FragColor=vec4(dFdx(gl_FragCoord.x),dFdy(gl_FragCoord.y),0.0,1.0);}");
    const camera = serializeCamera(new THREE.PerspectiveCamera(45, 32 / 31, .1, 10));

    const renderer = new CpuRasterizer(base.scene, camera,
      { ...settings, tileSize: 9, maxSamples: 4, dofEnabled: true, dofFocusDistance: 2, dofAperture: 1 },
      32, 31, new CpuEnvironment());

    const rendered = image(renderer);
    const offset = (15 * 32 + 16) * 4;
    expect(Array.from(rendered.slice(offset, offset + 4))).toEqual([255, 255, 0, 255]);
  });

  it("does not use an occluding primitive as the derivative neighbor", async() => {
    const renderer = await rendererFor(`varying vec2 vUv;
void main(){gl_FragColor=vec4(dFdx(vUv.x)*32.0,0.0,0.0,1.0);}`, 9);

    const scene = new THREE.Scene();
    const cover = new THREE.Mesh(new THREE.PlaneGeometry(1, 2), new THREE.MeshBasicMaterial({ color: 0x00ff00 }));
    cover.position.set(.4, 0, -1);
    scene.add(cover);
    const snapshot = (await serializeScene(scene)).scene;
    renderer.scene.meshes.push(...snapshot.meshes);
    const rendered = image(renderer);
    expect(Array.from(rendered.slice((15 * 32 + 13) * 4, (15 * 32 + 13) * 4 + 4))).toEqual([255, 0, 0, 255]);
    expect(Array.from(rendered.slice((15 * 32 + 14) * 4, (15 * 32 + 14) * 4 + 4))).toEqual([0, 255, 0, 255]);
  });

  it("supports material hooks before and after native shading", async() => {
    const material = new THREE.MeshBasicMaterial();

    material.onBeforeCompile = shader=>{
      shader.fragmentShader = shader.fragmentShader.replace("#include <color_fragment>",
        "#include <color_fragment>\ndiffuseColor.rgb=vec3(dFdx(vViewPosition.x)*-16.0);")
        .replace("#include <dithering_fragment>",
          "#include <dithering_fragment>\ngl_FragColor.g=abs(dFdy(vViewPosition.y))*15.5;");
    };

    const rendered = image(await rendererFor("",9,material));
    expect(rendered.every(v=>v === 255)).toBe(true);
  });
});
