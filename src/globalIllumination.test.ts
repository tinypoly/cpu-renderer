import { describe, expect, it } from "vitest";
import * as THREE from "three";
import { CpuRasterizer } from "./rasterizer.js";
import { CpuEnvironment } from "./environment.js";
import { ShadowBvh } from "./bvh.js";
import { serializeScene } from "./sceneSerialization.js";
import type { SerializedTexture } from "./sceneSerialization.js";
import type { FrameSettings } from "./frameSettings.js";

const settings: FrameSettings = {
  renderScale: 1, maxSamples: 1, tileSize: 8, shadows: true, environmentIntensity: 0,
  environmentRotation: 0, backgroundMode: "transparent", backgroundColor: "#000000",
  dofEnabled: false, dofFocusDistance: 3, dofAperture: 2.8, bokehBlades: 0,
  tonemapping: "linear", exposure: 1, globalIllumination: true, giSamples: 128, giBounces: 3,
};

const environment = new CpuEnvironment();
environment.sample = () => [1, 1, 1];
const camera = new THREE.PerspectiveCamera(45, 1, .1, 100);
camera.updateMatrixWorld();
const cameraState = { matrixWorld: camera.matrixWorld.toArray(), fov: 45, near: .1, far: 100 };

function plane(z: number, material: THREE.Material, size = 8) {
  const mesh = new THREE.Mesh(new THREE.PlaneGeometry(size, size), material);
  mesh.position.z = z;
  mesh.castShadow = mesh.receiveShadow = true;

  return mesh;
}

function receiver(color = .8) {
  return plane(-3, new THREE.MeshLambertMaterial({ color: new THREE.Color(color, color, color) }));
}

function emitter(size = 4, intensity = 2) {
  const mesh = plane(1, new THREE.MeshStandardMaterial({ color: 0, emissive: 0xff0000,
    emissiveIntensity: intensity }), size);

  mesh.rotation.y = Math.PI;

  return mesh;
}

async function render(objects: THREE.Object3D[], options: Partial<FrameSettings> = {}, width = 8) {
  const scene = new THREE.Scene(); scene.add(...objects);
  const serialized = (await serializeScene(scene)).scene;

  const renderer = new CpuRasterizer(serialized, cameraState, { ...settings, ...options },
    width, width, environment, { linear: true });

  for (const _ of renderer.prepare()) { /* cooperative preparation */ }

  const pixels = new Float32Array(width * width * 4);

  for (const bucket of renderer.buckets) {
    const job = renderer.renderBucket(bucket);
    let step = job.next();
    while (!step.done) step = job.next();
    for (let y = 0; y < bucket.height; y++) pixels.set(
      step.value.linear!.subarray(y * bucket.width * 4, (y + 1) * bucket.width * 4),
      ((bucket.y + y) * width + bucket.x) * 4);
  }

  const mean = [0, 0, 0];
  for (let y = width / 2 - 1; y <= width / 2; y++)
    for (let x = width / 2 - 1; x <= width / 2; x++)
      for (let c = 0; c < 3; c++) mean[c] += pixels[(y * width + x) * 4 + c] / 4;

  return { mean, pixels };
}

describe("diffuse global illumination", () => {
  it("illuminates from emissive geometry behind the camera and remains opt-in", async() => {
    const objects = [receiver(), emitter()];
    expect((await render(objects, { globalIllumination: false })).mean).toEqual([0, 0, 0]);
    const { mean } = await render(objects);
    expect(mean[0]).toBeGreaterThan(.2);
    expect(mean[1]).toBe(0); expect(mean[2]).toBe(0);
  });
  it("transfers color from a directly lit wall, without emission", async() => {
    const wall = plane(-2, new THREE.MeshLambertMaterial({ color: 0xff0000 }));
    wall.position.x = 2; wall.rotation.y = -Math.PI / 2;
    const light = new THREE.DirectionalLight(0xffffff, 4);
    light.position.set(-5, 0, 0); light.castShadow = true;
    const objects = [receiver(), wall, light];
    expect((await render(objects, { globalIllumination: false })).mean).toEqual([0, 0, 0]);
    const { mean } = await render(objects);
    expect(mean[0]).toBeGreaterThan(.1);
    expect(mean[1]).toBe(0); expect(mean[2]).toBe(0);
  });
  it("preserves unit environment radiance in an open hemisphere without double-counting AO", async() => {
    const { mean } = await render([receiver()], { environmentIntensity: 1, ambientOcclusion: true });
    for (const c of mean) expect(c).toBeCloseTo(.8, 5);
  });
  it("does not leak environment or ambient fill into a closed room, even without shadow flags", async() => {
    const room = new THREE.Mesh(new THREE.BoxGeometry(8, 8, 8),
      new THREE.MeshLambertMaterial({ color: 0xffffff, side: THREE.BackSide }));

    const { mean } = await render([room, new THREE.AmbientLight(0xffffff, Math.PI)],
      { environmentIntensity: 1, giBounces: 8, shadows: false });

    expect(mean).toEqual([0, 0, 0]);
  });
  it("blocks emissive light with non-shadow-casting geometry and allows alpha cutouts", async() => {
    const blocker = plane(.5, new THREE.MeshLambertMaterial({ color: 0 }));
    blocker.castShadow = false;
    const objects = [receiver(), emitter(), blocker];
    expect((await render(objects)).mean).toEqual([0, 0, 0]);
    (blocker.material as THREE.MeshLambertMaterial).opacity = 0;
    (blocker.material as THREE.MeshLambertMaterial).alphaTest = .5;
    expect((await render(objects)).mean[0]).toBeGreaterThan(.2);
  });
  it("adds energy with additional bounces in a diffuse enclosure", async() => {
    const room = new THREE.Mesh(new THREE.BoxGeometry(8, 8, 8),
      new THREE.MeshLambertMaterial({ color: new THREE.Color(.7, .7, .7), side: THREE.BackSide }));

    room.castShadow = room.receiveShadow = true;
    const objects = [room, emitter()];
    const one = (await render(objects, { giBounces: 1 })).mean[0];
    const three = (await render(objects, { giBounces: 3 })).mean[0];
    expect(one).toBeGreaterThan(0);
    expect(three).toBeGreaterThan(one * 1.1);
  });
  it("samples small emitters reliably without counting emission twice", async() => {
    // Small centered square: irradiance/pi approaches radiance * area / (pi * distance²).
    const { mean } = await render([receiver(1), emitter(.2, 20)], { giBounces: 1, giSamples: 256 });
    expect(mean[0]).toBeGreaterThan(.014);
    expect(mean[0]).toBeLessThan(.017);
    const large = (await render([receiver(1), emitter(200, 1)], { giBounces: 1, giSamples: 1024 })).mean[0];
    expect(large).toBeGreaterThan(.96);
    expect(large).toBeLessThan(1.04);
  });
  it("applies intensity once per path and clamps only indirect light", async() => {
    const objects = [receiver(), emitter()];
    const base = (await render(objects)).mean[0];
    const doubled = (await render(objects, { giIntensity: 2 })).mean[0];
    expect(doubled).toBeCloseTo(base * 2, 5);
    expect((await render(objects, { giClamp: .01 })).mean[0]).toBeLessThan(base * .2);
    expect((await render(objects, { giIntensity: 0 })).mean).toEqual([0, 0, 0]);
  });
  it("does not give perfect metals a diffuse GI lobe", async() => {
    const metal = plane(-3, new THREE.MeshStandardMaterial({ metalness: 1, color: 0xffffff }));
    expect((await render([metal, emitter()])).mean).toEqual([0, 0, 0]);
  });
  it("evaluates emissive textures at secondary hits", async() => {
    const light = emitter();
    const material = light.material as THREE.MeshStandardMaterial;
    material.emissive.setRGB(1, 1, 1);
    material.emissiveMap = new THREE.DataTexture(new Uint8Array([0, 255, 0, 255]), 1, 1);
    const { mean } = await render([receiver(), light]);
    expect(mean[0]).toBe(0);
    expect(mean[1]).toBeGreaterThan(.2);
    expect(mean[2]).toBe(0);
  });
  it("converges with more GI samples independently of antialiasing", async() => {
    const room = new THREE.Mesh(new THREE.BoxGeometry(8, 8, 8),
      new THREE.MeshLambertMaterial({ color: new THREE.Color(.7, .7, .7), side: THREE.BackSide }));

    const objects = [room, emitter()];
    const coarse = (await render(objects, { giSamples: 4 })).pixels;
    const fine = (await render(objects, { giSamples: 128 })).pixels;
    const reference = (await render(objects, { giSamples: 1024 })).pixels;
    const error = (pixels: Float32Array) => pixels.reduce((sum, v, i) => sum + (v - reference[i]) ** 2, 0);
    expect(error(fine)).toBeLessThan(error(coarse) * .2);
  });
  it("occludes indirect light even when direct-shadow flags are off", async() => {
    const room = new THREE.Mesh(new THREE.BoxGeometry(8, 8, 8),
      new THREE.MeshLambertMaterial({ color: 0xffffff, side: THREE.BackSide }));

    const light = new THREE.DirectionalLight(0xffffff, 4);
    light.position.set(0, 5, 0);
    // The visible back wall has n.l = 0; indirect light must not leak through the ceiling.
    expect((await render([room, light], { shadows: false })).mean).toEqual([0, 0, 0]);
  });
  it("is deterministic across buckets and keeps transparent backgrounds", async() => {
    const objects = [receiver(), emitter()];
    const a = await render(objects, { tileSize: 8 }, 16);
    const b = await render(objects, { tileSize: 16 }, 16);
    expect(a.pixels).toEqual(b.pixels);
    const empty = await render([], { environmentIntensity: 1 });
    expect(empty.pixels.every(v => v === 0)).toBe(true);
  });
  it.each([{ giSamples: NaN }, { giSamples: 1025 }, { giSamples: 1.5 }, { giBounces: 0 },
    { giBounces: 9 }, { giIntensity: Infinity }, { giClamp: -1 }])("rejects invalid budgets: %j", async options => {
    await expect(render([], options)).rejects.toThrow("invalid global illumination settings");
  });
});

/** Black equirectangular map with a single bright texel at the zenith: a sun diffuse rays rarely find. */
function sunTexture(width = 64, height = 32, radiance = 400): SerializedTexture {
  const data = new Float32Array(width * height * 4);
  for (let i = 0; i < width * height; i++) data[i * 4 + 3] = 1;
  // Column 48: the sun is on the +z side, in front of the receiver facing the camera.
  const x = 48, y = 2, i = (y * width + x) * 4;
  data[i] = data[i + 1] = data[i + 2] = radiance;

  return { id: "sun", width, height, data, colorSpace: "srgb-linear", wrapS: 1000, wrapT: 1001,
    matrix: [1, 0, 0, 0, 1, 0, 0, 0, 1], flipY: false, channel: 0, nearest: true };
}

describe("environment sampling by luminance", () => {
  it("exists only for peaked maps and integrates the density to one", () => {
    const sun = CpuEnvironment.fromTexture(sunTexture());
    expect(sun.importance).not.toBeNull();
    const flat = new Float32Array(64 * 32 * 4).fill(1);
    expect(CpuEnvironment.fromTexture({ ...sunTexture(), data: flat }).importance).toBeNull();
    // Monte Carlo over uniform directions: the mean of pdf times 4 pi must equal one.
    let sum = 0;
    const count = 200000;

    for (let i = 0; i < count; i++) {
      const z = 1 - 2 * ((i + .5) / count), r = Math.sqrt(1 - z * z), phi = i * 2.399963;
      sum += sun.pdf([r * Math.cos(phi), z, r * Math.sin(phi)], 0);
    }

    expect(sum / count * 4 * Math.PI).toBeCloseTo(1, 1);

    // The sampled direction lands where the density says it does, and rotation moves both together.
    for (const rotation of [0, 137]) {
      const picked = sun.sampleDirection(.3, .7, rotation)!;
      expect(picked.pdf).toBeGreaterThan(0);
      expect(sun.pdf(picked.direction, rotation)).toBeCloseTo(picked.pdf, 6);
      expect(sun.sample(picked.direction, rotation)[0]).toBeGreaterThan(0);
    }
  });
  it("removes the noise of a small sun without changing the energy", async() => {
    const sun = CpuEnvironment.fromTexture(sunTexture());
    const scene = new THREE.Scene(); scene.add(receiver(1));
    const serialized = (await serializeScene(scene)).scene;

    const image = (env: CpuEnvironment, samples: number) => {
      const renderer = new CpuRasterizer(serialized, cameraState, { ...settings, environmentIntensity: 1,
        giSamples: samples, giBounces: 1, giCache: false }, 8, 8, env, { linear: true });

      for (const _ of renderer.prepare()) { /* cooperative preparation */ }

      const values: number[] = [];

      for (const bucket of renderer.buckets) {
        const job = renderer.renderBucket(bucket);
        let step = job.next();
        while (!step.done) step = job.next();
        for (let i = 0; i < bucket.width * bucket.height; i++) values.push(step.value.linear![i * 4]);
      }

      const mean = values.reduce((a, b) => a + b, 0) / values.length;

      return { mean, deviation: Math.sqrt(values.reduce((a, v) => a + (v - mean) ** 2, 0) / values.length) };
    };

    const guided = image(sun, 16);
    // Analytic value: radiance times the texel solid angle times the cosine with the +z normal, over pi.
    const solidAngle = 2 * Math.PI / 64 * (Math.cos(2 / 32 * Math.PI) - Math.cos(3 / 32 * Math.PI));
    const cosine = Math.sin(2.5 / 32 * Math.PI) * Math.sin((48.5 / 64 - .5) * 2 * Math.PI);
    const expected = 400 * solidAngle * cosine / Math.PI;
    expect(guided.mean).toBeCloseTo(expected, 2);
    expect(guided.deviation).toBeLessThan(.002);
    // With diffuse rays alone, a sun this small is almost never found: either zero or a blown-out spot.
    const blind = CpuEnvironment.fromTexture(sunTexture());
    blind.importance = null;
    const unguided = image(blind, 16);
    expect(Math.abs(unguided.mean - expected)).toBeGreaterThan(Math.abs(guided.mean - expected) * 4);
  });
});
describe("irradiance cache and estimate", () => {
  it("returns finite variance and radius and matches brute force within noise", async() => {
    // A side-lit white wall bounces light onto the receiver, as in the color transfer test.
    const wall = plane(-2, new THREE.MeshLambertMaterial({ color: 0xffffff }));
    wall.position.x = 2; wall.rotation.y = -Math.PI / 2;
    const light = new THREE.DirectionalLight(0xffffff, 4);
    light.position.set(-5, 0, 0); light.castShadow = true;
    const objects = [receiver(.8), wall, light];
    const exact = await render(objects, { giCache: false, giSamples: 256 }, 16);
    const cached = await render(objects, { giCache: true, giCacheSpacing: 4, giSamples: 256 }, 16);
    for (let c = 0; c < 3; c++) expect(cached.mean[c]).toBeCloseTo(exact.mean[c], 1);
    expect(cached.mean[0]).toBeGreaterThan(.05);
    let cachedNoise = 0, exactNoise = 0;

    for (let i = 0; i < 16 * 16; i++) {
      cachedNoise += Math.abs(cached.pixels[i * 4] - cached.mean[0]);
      exactNoise += Math.abs(exact.pixels[i * 4] - exact.mean[0]);
    }

    // The cache interpolates: the image comes out at least as smooth as the exact one.
    expect(cachedNoise).toBeLessThanOrEqual(exactNoise * 1.2);
  });
});
describe("GI BVH closest hits", () => {
  it("returns the nearest surface, barycentrics and non-casters without changing shadow queries", () => {
    const triangle = (z: number, castShadow: boolean) => ({
      positions: [[-2, -2, z], [2, -2, z], [0, 2, z]], castShadow });

    const bvh = new ShadowBvh([triangle(-5, true), triangle(-2, false)], true);
    expect(bvh.intersect([0, 0, 0], [0, 0, -1])).toEqual({ index: 1, u: .25, v: .5, distance: 2 });
    expect(bvh.occluded([0, 0, 0], [0, 0, -1], 3, -1, () => true)).toBe(false);
    expect(bvh.intersect([3, 0, 0], [0, 0, -1])).toBeNull();
    expect(bvh.intersect([0, 0, 0], [0, 0, -1], 1)).toBeNull();
  });
});
describe("denoiser buffers", () => {
  it("recomposes the bucket image when the filtered light is the original itself", async() => {
    const { allocateFrameAov, denoiseIndirect } = await import("./denoise.js");
    const scene = new THREE.Scene(); scene.add(receiver(.8), emitter());
    const serialized = (await serializeScene(scene)).scene;
    const width = 16;

    const renderer = new CpuRasterizer(serialized, cameraState, { ...settings, tileSize: 8, maxSamples: 2,
      tonemapping: "aces", giSamples: 32, giCache: true }, width, width, environment);

    for (const _ of renderer.prepare()) { /* cooperative preparation */ }

    const aov = allocateFrameAov(width, width);
    renderer.attachAov(aov);
    const image = new Uint8ClampedArray(width * width * 4);

    for (const bucket of renderer.buckets) {
      const job = renderer.renderBucket(bucket);
      let step = job.next();
      while (!step.done) step = job.next();
      for (let y = 0; y < bucket.height; y++) image.set(
        step.value.pixels.subarray(y * bucket.width * 4, (y + 1) * bucket.width * 4),
        ((bucket.y + y) * width + bucket.x) * 4);
    }

    const center = (width / 2) * width + width / 2;
    expect(aov.depth[center]).toBeGreaterThan(0);
    // Modulation summed over the pixel's two antialiasing samples.
    expect(aov.modulation[center * 3]).toBeCloseTo(.8 * 2, 5);
    expect(aov.indirect[center * 3]).toBeGreaterThan(0);
    const same = new Uint8ClampedArray(width * width * 4);
    const identity = renderer.resolveDenoised(aov.indirect, (row, pixels) => same.set(pixels, row * width * 4));
    let step = identity.next();
    while (!step.done) step = identity.next();
    expect(Array.from(same)).toEqual(Array.from(image));
    // With the real filter the image stays finite, opaque where there is a surface and of the same brightness order.
    const job = denoiseIndirect(aov);
    let filtered = job.next();
    while (!filtered.done) filtered = job.next();
    const out = new Uint8ClampedArray(width * width * 4);
    const resolved = renderer.resolveDenoised(filtered.value, (row, pixels) => out.set(pixels, row * width * 4));
    step = resolved.next();
    while (!step.done) step = resolved.next();
    expect(out[center * 4 + 3]).toBe(255);
    expect(Math.abs(out[center * 4] - image[center * 4])).toBeLessThan(40);
  });
});
