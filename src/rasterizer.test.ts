import { describe, expect, it } from "vitest";
import * as THREE from "three";
import { CpuRasterizer } from "./rasterizer.js";
import { CpuEnvironment } from "./environment.js";
import { serializeScene } from "./sceneSerialization.js";
import type { FrameSettings } from "./frameSettings.js";
import type { VolumetricMedium } from "./volumetricLight.js";

const settings: FrameSettings = {
  renderScale: 1,
  maxSamples: 4,
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
  tonemapping: "aces",
  exposure: 1,
};

function triangle(color: number, z: number, transparent = false) {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute([-1, -1, z, 1, -1, z, 0, 1, z], 3));

  return new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({ color, transparent, opacity: transparent ? .5 : 1 }));
}

async function prepareRenderer(options: Partial<FrameSettings>) {
  const scene = new THREE.Scene();
  const back = triangle(0x00ff00, -3);
  back.position.x = .4;
  scene.add(back, triangle(0xff0000, -2), triangle(0x0000ff, -1.5, true));
  const { scene: serialized } = await serializeScene(scene);
  const camera = new THREE.PerspectiveCamera(60, 1, .1, 100);
  camera.updateMatrixWorld();

  const renderer = new CpuRasterizer(serialized, {
    matrixWorld: camera.matrixWorld.toArray(),
    fov: 60,
    near: .1,
    far: 100,
  }, { ...settings, ...options }, 16, 16, new CpuEnvironment());

  for (const _step of renderer.prepare()) { /* drain cooperative preparation */ }

  return renderer;
}

function drain(renderer: CpuRasterizer, progress?: (pixels: Uint8ClampedArray, pixel: number) => void) {
  const job = renderer.renderBucket(renderer.buckets[0], progress);
  let step = job.next();
  while (!step.done)
    step = job.next();

  return step.value;
}

describe("progressive bucket output", () => {
  it("delivers pixels in row order and ends identical to the bucket without progress", async() => {
    const renderer = await prepareRenderer({});
    const order: number[] = [];
    let live: Uint8ClampedArray | null = null;

    const result = drain(renderer, (pixels, pixel) => {
      order.push(pixel);
      live = pixels;
    });

    expect(order).toEqual(Array.from({ length: 256 }, (_, pixel) => pixel));
    expect(Array.from(live!)).toEqual(Array.from(result.pixels));
    expect(Array.from(result.pixels)).toEqual(Array.from(drain(renderer).pixels));
    expect(result.pixels.some(v => v > 0)).toBe(true);
  });

  it("with depth of field repeats the row order on every lens pass", async() => {
    const renderer = await prepareRenderer({ dofEnabled: true, maxSamples: 3, dofAperture: .5 });
    const order: number[] = [];
    const result = drain(renderer, (_pixels, pixel) => order.push(pixel));
    const pass = Array.from({ length: 256 }, (_, pixel) => pixel);

    expect(order).toEqual([...pass, ...pass, ...pass]);
    expect(Array.from(result.pixels)).toEqual(Array.from(drain(renderer).pixels));
  });
});

describe("irradiance cache preview", () => {
  it("shows records as dots before the rows and ends with the same image", async() => {
    const scene = new THREE.Scene();
    const floor = new THREE.Mesh(new THREE.PlaneGeometry(6, 6), new THREE.MeshLambertMaterial({ color: 0xffffff }));
    floor.position.z = -3;
    scene.add(floor);
    const { scene: serialized } = await serializeScene(scene);
    const camera = new THREE.PerspectiveCamera(60, 1, .1, 100);
    camera.updateMatrixWorld();
    const environment = new CpuEnvironment();
    environment.sample = () => [1, 1, 1];

    const renderer = new CpuRasterizer(serialized, {
      matrixWorld: camera.matrixWorld.toArray(), fov: 60, near: .1, far: 100,
    }, { ...settings, maxSamples: 1, globalIllumination: true, giSamples: 4, giBounces: 1, giCache: true,
      giCacheSpacing: 4, environmentIntensity: 1 }, 16, 16, environment);

    for (const _step of renderer.prepare()) { /* drain cooperative preparation */ }

    const order: number[] = [], alphas: number[] = [];

    const result = drain(renderer, (pixels, pixel) => {
      order.push(pixel);
      alphas.push(pixels[pixel * 4 + 3]);
    });

    // The first notifications are the grid dots (2x2 pixels every 4), out of row order and already opaque.
    expect(order.slice(0, 4)).toEqual([0, 1, 16, 17]);
    expect(alphas.slice(0, 4)).toEqual([255, 255, 255, 255]);
    const dots = 16 * 4;
    expect(order.slice(dots)).toEqual(Array.from({ length: 256 }, (_, pixel) => pixel));
    expect(Array.from(result.pixels)).toEqual(Array.from(drain(renderer).pixels));
  });
});

describe("full-image cache prepass", () => {
  it("fills the grid by ray across levels, only where missing, and shading matches the exact result", async() => {
    const { allocateIrradianceGrid } = await import("./irradianceCache.js");
    const scene = new THREE.Scene();
    const floor = new THREE.Mesh(new THREE.PlaneGeometry(6, 6), new THREE.MeshLambertMaterial({ color: 0xffffff }));
    floor.position.z = -3;
    const box = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), new THREE.MeshLambertMaterial({ color: 0xff4444 }));
    box.position.set(.6, 0, -2.4);
    scene.add(floor, box);
    const { scene: serialized } = await serializeScene(scene);
    const camera = new THREE.PerspectiveCamera(60, 1, .1, 100);
    camera.updateMatrixWorld();
    const environment = new CpuEnvironment();
    environment.sample = () => [1, 1, 1];
    const width = 32;

    const make = (cache: boolean) => {
      const renderer = new CpuRasterizer(serialized, {
        matrixWorld: camera.matrixWorld.toArray(), fov: 60, near: .1, far: 100,
      }, { ...settings, maxSamples: 1, globalIllumination: true, giSamples: 64, giBounces: 1, giCache: cache,
        giCacheSpacing: 2, environmentIntensity: 1 }, width, width, environment);

      for (const _step of renderer.prepare()) { /* drain cooperative preparation */ }

      return renderer;
    };

    const renderer = make(true);
    const grid = allocateIrradianceGrid(width, width, 2);
    renderer.attachIrradianceGrid(grid);
    const jobs = CpuRasterizer.prepassJobs(width, width, 2);
    expect(jobs.map(level => level[0].level)).toEqual([4, 2, 1]);
    const perLevel: number[] = [], dots: number[] = [];
    let index = 0;

    for (const level of jobs) {
      let written = 0;

      for (const _job of level) {
        const job = renderer.cachePrepass(index++, (_pixels, pixel) => dots.push(pixel));
        let step = job.next();
        while (!step.done) step = job.next();
        written += step.value.length;
      }

      perLevel.push(written);
    }

    // The coarse level covers the image; finer levels only add where the flat floor does not reach (box corner).
    expect(perLevel[0]).toBeGreaterThan(0);
    expect(perLevel[2]).toBeLessThan((width / 2) ** 2 / 2);
    expect(dots.length).toBeGreaterThan(0);
    expect(grid.flags.reduce((a, b) => a + b, 0)).toBe(perLevel.reduce((a, b) => a + b, 0));
    // Shading with the grid: rows in order only, and the image stays close to the exact result.
    const cached = new Uint8ClampedArray(width * width * 4), exact = new Uint8ClampedArray(width * width * 4);
    const plain = make(false);

    for (const [target, out] of [[renderer, cached], [plain, exact]] as const)
      for (const bucket of target.buckets) {
        const order: number[] = [];
        const job = target.renderBucket(bucket, (_pixels, pixel) => order.push(pixel));
        let step = job.next();
        while (!step.done) step = job.next();
        expect(order).toEqual(Array.from({ length: bucket.width * bucket.height }, (_, pixel) => pixel));
        for (let y = 0; y < bucket.height; y++)
          out.set(
            step.value.pixels.subarray(y * bucket.width * 4, (y + 1) * bucket.width * 4),
            ((bucket.y + y) * width + bucket.x) * 4);
      }

    let difference = 0, count = 0;

    for (let i = 0; i < width * width; i++) {
      if (exact[i * 4 + 3] === 0) continue;
      difference += Math.abs(cached[i * 4] - exact[i * 4]); count++;
    }

    expect(count).toBeGreaterThan(0);
    expect(difference / count).toBeLessThan(6);
    // Without the cache there are no grid jobs.
    const none = plain.cachePrepass(0);
    let noneStep = none.next();
    while (!noneStep.done) noneStep = none.next();
    expect(noneStep.value).toEqual([]);
  });
});

describe("volumetric lighting", () => {
  async function renderVolume(enabled: boolean | VolumetricMedium, blocker = false, spot = false, away = false,
    opaque = false, fog: THREE.FogExp2 | null = null) {
    const scene = new THREE.Scene();
    scene.fog = fog;
    const light = spot ? new THREE.SpotLight(0xff0000, 10, 4, .3, .2) : new THREE.PointLight(0xff0000, 10, 2);
    light.position.set(0, 0, -3);
    light.userData.cpuRenderer = { volumetric: enabled };

    if (light instanceof THREE.SpotLight) {
      light.target.position.set(away ? 10 : 0, 0, away ? -3 : 0);
      scene.add(light.target);
    }

    scene.add(light);

    if (blocker) {
      const plane = new THREE.Mesh(new THREE.PlaneGeometry(100, 100), new THREE.MeshBasicMaterial({ color: 0 }));
      plane.position.z = -.5;
      scene.add(plane);
    }

    const { scene: serialized } = await serializeScene(scene);
    expect(serialized.lights[0].volumetric).toBe(enabled !== false);

    const renderer = new CpuRasterizer(serialized, {
      matrixWorld: new THREE.Matrix4().toArray(), fov: 60, near: .1, far: 100,
    }, { ...settings, maxSamples: 1, backgroundMode: opaque ? "color" : "transparent" }, 8, 8, new CpuEnvironment());

    for (const _step of renderer.prepare()) { /* drain */ }

    return drain(renderer).pixels;
  }

  it("renders glow and alpha over a transparent background only when enabled", async() => {
    const enabled = await renderVolume(true), disabled = await renderVolume(false);
    expect(disabled.every(v => v === 0)).toBe(true);
    const center = (4 * 8 + 4) * 4;
    expect(enabled[center]).toBeGreaterThan(0);
    expect(enabled[center]).toBeGreaterThan(enabled[center + 1]);
    expect(enabled[center + 3]).toBeGreaterThan(0);
  });

  it("preserves the visible glow when compositing the transparent background over black", async() => {
    const transparent = await renderVolume(true), opaque = await renderVolume(true, false, false, false, true);

    for (let i = 0; i < transparent.length; i++)
      if (i % 4 !== 3) {
        const alpha = transparent[i - i % 4 + 3] / 255;
        expect(Math.abs(transparent[i] * alpha - opaque[i])).toBeLessThanOrEqual(1);
      }
  });

  it("does not show the volume behind an opaque surface", async() => {
    const pixels = await renderVolume(true, true);
    expect(pixels.every((v, i) => i % 4 === 3 ? v === 255 : v === 0)).toBe(true);
  });

  it("dims the glow through the scene fog in front of it", async() => {
    const clear = await renderVolume(true, false, false, false, true);
    const fogged = await renderVolume(true, false, false, false, true, new THREE.FogExp2(0x000000, .4));

    const center = (4 * 8 + 4) * 4;
    expect(fogged[center]).toBeGreaterThan(0);
    expect(fogged[center]).toBeLessThan(clear[center] * .7);
  });

  it("reads the medium from userData.cpuRenderer.volumetric", async() => {
    const light = new THREE.PointLight();
    light.userData.cpuRenderer = { volumetric: { density: .05, anisotropy: .6, spread: Number.NaN } };
    const { scene } = await serializeScene(new THREE.Scene().add(light));
    expect(scene.lights[0]).toMatchObject({ volumetric: true, medium: { density: .05, anisotropy: .6 } });
    expect(scene.lights[0].medium).not.toHaveProperty("spread");
  });

  it("scatters forward toward the camera with a positive anisotropy", async() => {
    const center = (4 * 8 + 4) * 4;
    const forward = await renderVolume({ anisotropy: .6 }, false, false, false, true);
    const backward = await renderVolume({ anisotropy: -.6 }, false, false, false, true);
    expect(forward[center]).toBeGreaterThan(backward[center]);
  });

  it("glows brighter in denser air and spreads past the cone", async() => {
    const total = (pixels: ArrayLike<number>) => Array.from(pixels).reduce((sum, v, i) => i % 4 ? sum : sum + v, 0);
    const thin = await renderVolume(true, false, true, false, true);
    const dense = await renderVolume({ density: .03 }, false, true, false, true);
    const spread = await renderVolume({ spread: 1 }, false, true, false, true);
    expect(total(dense)).toBeGreaterThan(total(thin));
    expect(total(spread)).toBeGreaterThan(total(thin));
  });

  it("respects the spot light direction and cone", async() => {
    const toward = await renderVolume(true, false, true), away = await renderVolume(true, false, true, true);
    const center = (4 * 8 + 4) * 4;
    expect(toward[center + 3]).toBeGreaterThan(away[center + 3]);
  });
});
