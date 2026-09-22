import { afterEach, describe, expect, it, vi } from "vitest";
import * as THREE from "three";
import { CpuRasterizer } from "./rasterizer.js";
import { CpuEnvironment } from "./environment.js";
import { matrixValue } from "./glsl.js";
import { interpolateStored, readValue, valueShape, writeValue } from "./preparedGeometry.js";
import { serializeScene } from "./sceneSerialization.js";
import type { FrameSettings } from "./frameSettings.js";

const settings: FrameSettings = {
  renderScale: 1, maxSamples: 2, tileSize: 16, shadows: true, environmentIntensity: 1, environmentRotation: 30,
  backgroundMode: "environment", backgroundColor: "#000000", dofEnabled: false, dofFocusDistance: 5,
  dofAperture: 2.8, bokehBlades: 0, tonemapping: "aces", exposure: 1,
};

const camera = new THREE.PerspectiveCamera(50, 32 / 24, .1, 100);
camera.position.set(2, 2.5, 5);
camera.lookAt(0, 0, 0);
camera.updateMatrixWorld();
const cameraState = { matrixWorld: camera.matrixWorld.toArray(), fov: 50, near: .1, far: 100 };

afterEach(() => vi.unstubAllGlobals());

function drain<T>(job: Generator<void, T>): T {
  let step = job.next();
  while (!step.done)
    step = job.next();

  return step.value;
}

function image(renderer: CpuRasterizer) {
  drain(renderer.prepare());

  return renderer.buckets.map(bucket => Array.from(drain(renderer.renderBucket(bucket)).pixels));
}

const gradient = () => CpuEnvironment.load({ kind: "gradient", topColor: "#bcd4ff", bottomColor: "#302820", exponent: 1 });

/** One worker prepares; another receives the preparation through postMessage structured clone and only renders. */
async function expectAdoptedImage(objects: THREE.Object3D[], options: Partial<FrameSettings>) {
  const scene = new THREE.Scene();
  scene.add(...objects);
  const serialized = (await serializeScene(scene)).scene, environment = await gradient();
  const merged = { ...settings, ...options };
  const owner = new CpuRasterizer(serialized, cameraState, merged, 32, 24, environment);
  const expected = image(owner);

  const frame = structuredClone({
    scene: serialized, environment: environment.snapshot(), geometry: owner.preparedGeometry,
  });

  const reader = new CpuRasterizer(frame.scene, cameraState, merged, 32, 24,
    CpuEnvironment.fromSnapshot(frame.environment), {}, frame.geometry);

  expect(image(reader)).toEqual(expected);
  expect(expected.some(bucket => bucket.some(value => value > 0))).toBe(true);
}

function texture(pixels: number[]) {
  const map = new THREE.DataTexture(new Uint8Array(pixels), 2, 2, THREE.RGBAFormat);
  map.needsUpdate = true;

  return map;
}

function litObjects() {
  const floor = new THREE.Mesh(new THREE.PlaneGeometry(8, 8).rotateX(-Math.PI / 2),
    new THREE.MeshStandardMaterial({ color: 0xb0a090, roughness: .6,
      normalMap: texture([128, 150, 255, 255, 110, 128, 255, 255, 128, 128, 255, 255, 140, 120, 255, 255]) }));

  floor.receiveShadow = true;

  const box = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1),
    new THREE.MeshLambertMaterial({ color: 0x3080ff, flatShading: true }));

  box.position.y = .5;
  box.castShadow = box.receiveShadow = true;
  // No normals: shading uses the face normal.
  const loose = new THREE.BufferGeometry();
  loose.setAttribute("position", new THREE.Float32BufferAttribute([-2, 0, 1, -1, 0, 1, -1.5, 1.2, 1], 3));
  const shard = new THREE.Mesh(loose, new THREE.MeshPhongMaterial({ color: 0xff8040, side: THREE.DoubleSide }));
  shard.castShadow = true;

  // Alpha cutout: the shadow ray samples the texture instead of accepting the hit outright.
  const leaf = new THREE.Mesh(new THREE.PlaneGeometry(1.2, 1.2), new THREE.MeshStandardMaterial({ alphaTest: .5,
    map: texture([255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255]), side: THREE.DoubleSide }));

  leaf.position.set(1.2, 1.4, 0);
  leaf.castShadow = true;
  const sun = new THREE.DirectionalLight(0xffffff, 2.5);
  sun.position.set(3, 5, 2);

  return [floor, box, shard, leaf, sun, new THREE.AmbientLight(0xffffff, .2)];
}

describe("preparation shared between workers", () => {
  it("renders the same image with soft shadows, occlusion, fog, face normals and cutouts", async() => {
    await expectAdoptedImage(litObjects(),
      { ambientOcclusion: true, shadowSoftness: 2, fogEnabled: true, fogNear: 2, fogFar: 12 });
  });

  it("renders the same image with global illumination and emitters", async() => {
    const panel = new THREE.Mesh(new THREE.PlaneGeometry(2, 2),
      new THREE.MeshStandardMaterial({ color: 0, emissive: 0xffaa66, emissiveIntensity: 3 }));

    panel.position.set(-1.5, 1.5, -1);
    await expectAdoptedImage([...litObjects(), panel], { globalIllumination: true, giSamples: 4, giBounces: 2 });
  });

  it("renders the same image with shader varyings, points and depth of field", async() => {
    const card = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), new THREE.ShaderMaterial({
      vertexShader: "varying vec3 vTint;varying float vHeight;void main(){vTint=vec3(uv,.5);vHeight=position.y;"
        + "gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0);}",
      fragmentShader: "varying vec3 vTint;varying float vHeight;void main(){gl_FragColor=vec4(vTint*(.6+vHeight*.3),.7);}",
      transparent: true, side: THREE.DoubleSide,
    }));

    const sprites = new THREE.BufferGeometry();
    sprites.setAttribute("position", new THREE.Float32BufferAttribute([0, 1, 1, .5, 1.5, 1, -.6, .8, 1.5], 3));

    const points = new THREE.Points(sprites, new THREE.ShaderMaterial({
      vertexShader: "void main(){gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0);gl_PointSize=6.0;}",
      fragmentShader: "void main(){if(length(gl_PointCoord-0.5)>0.5)discard;gl_FragColor=vec4(gl_PointCoord,1.0,1.0);}",
    }));

    await expectAdoptedImage([card, points], { dofEnabled: true, dofAperture: .8, maxSamples: 3, bokehBlades: 5 });
  });

  it("stores whole matrix varyings from the first vertex, like the object-based interpolation", () => {
    const a = matrixValue([1, 2, 3, 4]), b = matrixValue([5, 6, 7, 8]), shape = valueShape(a);
    const data = new Float64Array(10);
    expect(writeValue(a, shape, data, 0)).toBe(5);
    writeValue(b, shape, data, 5);
    expect(readValue(data, 5, shape, [0])).toEqual(b);
    expect(interpolateStored(data, 0, 5, 5, shape, .2, .4, .4, [0])).toEqual(a);
    expect(() => writeValue([1, 2], shape, data, 0)).toThrow("changed type");
  });

  it("publishes geometry, BVH and environment in shared memory when the page is isolated", async() => {
    vi.stubGlobal("crossOriginIsolated", true);
    const scene = new THREE.Scene();
    scene.add(...litObjects());
    const environment = await gradient();
    const owner = new CpuRasterizer((await serializeScene(scene)).scene, cameraState, settings, 32, 24, environment);
    drain(owner.prepare());
    const { vertices, triangles, positions, binTriangles, bvh } = owner.preparedGeometry;
    for (const array of [vertices, triangles, positions, binTriangles, bvh.bounds, bvh.nodes, bvh.indices])
      expect(array.buffer).toBeInstanceOf(SharedArrayBuffer);
    expect(environment.snapshot().levels.every(level => level.texels.buffer instanceof SharedArrayBuffer)).toBe(true);
  });
});
