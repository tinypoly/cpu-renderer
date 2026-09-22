import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import * as THREE from "three";
import { serializeCamera } from "./camera.js";
import { shareable } from "./sharedBuffers.js";
import {
  CpuRenderer,
  defaultWorkerCount,
  effectiveWorkerCount,
  isAbortError,
  MAX_RENDER_WORKERS,
  type CpuRendererOptions,
  type RenderPixels,
  type RenderWorker,
} from "./renderer.js";
import type { WorkerRequest, WorkerResponse } from "./protocol.js";
import type { SerializedSceneResult } from "./sceneSerialization.js";

class FakeWorker {
  static instances: FakeWorker[] = [];
  postMessage = vi.fn<(message: WorkerRequest, transfer?: Transferable[]) => void>();
  terminate = vi.fn();
  onmessage: ((event: MessageEvent<WorkerResponse>) => void) | null = null;

  constructor() {
    FakeWorker.instances.push(this);
  }

  /** The worker answers: delivers a response as if it came over the channel. */
  reply(message: WorkerResponse) {
    this.onmessage?.({ data: message } as MessageEvent<WorkerResponse>);
  }

  /** The last request of one type, with its revision. */
  last<T extends WorkerRequest["type"]>(type: T): Extract<WorkerRequest, { type: T }> | undefined {
    return this.postMessage.mock.calls.map(call => call[0]).filter(message => message.type === type).at(-1) as
      Extract<WorkerRequest, { type: T }> | undefined;
  }
}

const emptyScene = (): SerializedSceneResult => ({ scene: { meshes: [], lights: [], textures: [] }, transfer: [] });
const createWorker = () => new FakeWorker() as unknown as RenderWorker;

/** A renderer on fake workers, 32 × 32 unless the options say otherwise. */
const create = (options: Partial<CpuRendererOptions> = {}) =>
  new CpuRenderer({ width: 32, height: 32, createWorker, ...options });

/** Plays the owner worker through a frame of two buckets, without denoiser or cache. */
function completeFrame(worker: FakeWorker, revision: number) {
  worker.reply({ type: "start", revision, width: 32, height: 32, order: [0, 1], denoise: false, prepass: [] });
  const pixels = new Uint8ClampedArray(16 * 32 * 4);
  worker.reply({ type: "bucket", revision, bucket: { x: 0, y: 0, width: 32, height: 16, index: 0 }, pixels });
  worker.reply({ type: "bucket", revision, bucket: { x: 0, y: 16, width: 32, height: 16, index: 1 }, pixels });
}

/** Lets `render()` await the scene and post the frame. */
const settle = () => new Promise<void>(resolve => setTimeout(resolve, 0));

beforeEach(() => {
  FakeWorker.instances = [];
});
afterEach(() => vi.unstubAllGlobals());

describe("worker pool size", () => {
  it("keeps one core free and caps the pool", () => {
    expect(defaultWorkerCount()).toBeGreaterThanOrEqual(1);
    expect(defaultWorkerCount()).toBeLessThanOrEqual(MAX_RENDER_WORKERS);
    vi.stubGlobal("crossOriginIsolated", true);
    expect(defaultWorkerCount(16)).toBe(8);
    expect(defaultWorkerCount(4)).toBe(3);
    expect(defaultWorkerCount(1)).toBe(1);
  });
  it("shares memory in Node without cross-origin isolation, unless the page says otherwise", () => {
    vi.unstubAllGlobals();
    expect(shareable()).toBe(true);
    vi.stubGlobal("crossOriginIsolated", false);
    expect(shareable()).toBe(false);
  });
  it("bounds automatic concurrency and avoids scene copies without shared memory", () => {
    vi.stubGlobal("navigator", { hardwareConcurrency: 32 });
    vi.stubGlobal("crossOriginIsolated", true);
    expect(defaultWorkerCount()).toBe(8);
    expect(effectiveWorkerCount(100)).toBe(MAX_RENDER_WORKERS);
    vi.stubGlobal("crossOriginIsolated", false);
    expect(defaultWorkerCount()).toBe(1);
    expect(effectiveWorkerCount(16)).toBe(1);
  });
});

describe("CpuRenderer", () => {
  it.each([false, true])("requests a published frame only when readers exist (shared=%s)", async shared => {
    vi.stubGlobal("crossOriginIsolated", shared);
    const renderer = create({ workers: 4 });
    expect(renderer.workerCount).toBe(shared ? 4 : 1);
    void renderer.render().catch(() => {});
    await settle();
    expect(FakeWorker.instances[0].last("frame")).toMatchObject({ publishPrepared: shared });

    // Readers open the revision with a reset and never receive the frame itself.
    for (const reader of FakeWorker.instances.slice(1)) {
      expect(reader.last("frame")).toBeUndefined();
      expect(reader.last("reset")).toBeDefined();
    }

    renderer.dispose();
    expect(FakeWorker.instances.every(worker => worker.terminate.mock.calls.length === 1)).toBe(true);
  });

  it("sends every input on the first frame and only the changed ones afterwards", async() => {
    vi.stubGlobal("crossOriginIsolated", false);

    const renderer = create({
      settings: { samples: 1 },
      environment: { kind: "gradient", topColor: "#fff", bottomColor: "#000" },
      width: 64, height: 48, pixelRatio: 2,
    });

    const [worker] = FakeWorker.instances;
    const first = renderer.render();
    await settle();
    const frame = worker.last("frame")!;
    expect(frame).toMatchObject({ width: 64, height: 48, pixelRatio: 2, settings: { maxSamples: 1, tileSize: 64 } });
    expect(frame.scene).toEqual({ meshes: [], lights: [], textures: [] });
    expect(frame.environment).toEqual({ kind: "gradient", topColor: "#fff", bottomColor: "#000" });
    expect(frame.camera.matrixWorld).toHaveLength(16);
    completeFrame(worker, frame.revision);
    // The promise carries the finished image, the same buffer `image` holds.
    const image = await first;
    expect([image.width, image.height, image.data.length]).toEqual([32, 32, 32 * 32 * 4]);
    expect(image).toBe(renderer.image);

    // Nothing changed: the frame restarts with the same inputs, without resending the scene or the environment.
    renderer.setSettings({ samples: 8 });
    const second = renderer.render();
    await settle();
    const again = worker.last("frame")!;
    expect(again.revision).toBe(frame.revision + 1);
    expect(again.scene).toBeUndefined();
    expect(again.environment).toBeUndefined();
    expect(again.settings.maxSamples).toBe(8);
    expect(renderer.settings.samples).toBe(8);
    completeFrame(worker, again.revision);
    await expect(second).resolves.toMatchObject({ width: 32, height: 32 });

    renderer.setScene(emptyScene());
    renderer.setEnvironment({ kind: "hdri", url: "studio.hdr" });
    const third = renderer.render();
    await settle();
    const changed = worker.last("frame")!;
    expect(changed.scene).toBeDefined();
    expect(changed.environment).toEqual({ kind: "hdri", url: "studio.hdr" });
    renderer.dispose();
    await expect(third).rejects.toSatisfy(isAbortError);
  });

  it("resolves render() when the last bucket is painted and reports progress through events", async() => {
    vi.stubGlobal("crossOriginIsolated", false);
    const renderer = create();
    const [worker] = FakeWorker.instances;
    const progress: unknown[] = [], events: string[] = [];
    renderer.on("progress", next => progress.push(next));
    renderer.on("start", () => events.push("start"));
    renderer.on("complete", () => events.push("complete"));
    const done = renderer.render();
    expect(renderer.rendering).toBe(true);
    await settle();
    const { revision } = worker.last("frame")!;
    expect(progress.at(-1)).toMatchObject({ phase: "prepare" });
    worker.reply({ type: "start", revision, width: 32, height: 32, order: [1, 0], denoise: false, prepass: [] });
    // The first bucket in the announced order goes to the worker as soon as it is ready.
    expect(worker.last("render")).toMatchObject({ bucket: 1, phase: "shade" });
    const pixels = new Uint8ClampedArray(16 * 32 * 4);
    worker.reply({ type: "bucket", revision, bucket: { x: 0, y: 16, width: 32, height: 16, index: 1 }, pixels });
    expect(worker.last("render")).toMatchObject({ bucket: 0 });
    expect(progress.at(-1)).toMatchObject({ completed: 1, total: 2, phase: "shade" });
    expect(events).toEqual(["start"]);
    worker.reply({ type: "bucket", revision, bucket: { x: 0, y: 0, width: 32, height: 16, index: 0 }, pixels });
    await expect(done).resolves.toBe(renderer.image);
    expect(events).toEqual(["start", "complete"]);
    expect(renderer.rendering).toBe(false);
    // Output of a finished revision is ignored.
    worker.reply({ type: "error", revision, message: "late" });
    expect(renderer.rendering).toBe(false);
  });

  it("aborts the frame in flight when render() is called again, and fails it on a worker error", async() => {
    vi.stubGlobal("crossOriginIsolated", false);
    const renderer = create();
    const [worker] = FakeWorker.instances;
    const errors: Error[] = [];
    renderer.on("error", error => errors.push(error));
    const first = renderer.render();
    await settle();
    const second = renderer.render();
    await expect(first).rejects.toSatisfy(isAbortError);
    await settle();
    const { revision } = worker.last("frame")!;
    worker.reply({ type: "error", revision, message: "Unsupported shader syntax." });
    await expect(second).rejects.toThrow("Unsupported shader syntax.");
    expect(errors.map(error => error.message)).toEqual(["Unsupported shader syntax."]);
    // Two calls while the scene is still settling: only the last one posts a frame.
    renderer.setScene(emptyScene());
    const third = renderer.render(), fourth = renderer.render();
    await expect(third).rejects.toSatisfy(isAbortError);
    await settle();
    expect(worker.postMessage.mock.calls.filter(call => call[0].type === "frame")).toHaveLength(3);
    renderer.dispose();
    await expect(fourth).rejects.toSatisfy(isAbortError);
    await expect(renderer.render()).rejects.toThrow("disposed");
  });

  it("pauses and resumes every worker without opening a revision", async() => {
    vi.stubGlobal("crossOriginIsolated", true);
    const renderer = create({ workers: 2 });
    void renderer.render().catch(() => {});
    await settle();
    const revision = FakeWorker.instances[0].last("frame")!.revision;
    renderer.pause();
    renderer.pause();
    expect(renderer.paused).toBe(true);
    renderer.resume();

    for (const worker of FakeWorker.instances) {
      const pauses = worker.postMessage.mock.calls.map(call => call[0]).filter(message => message.type === "pause");
      expect(pauses).toEqual([{ type: "pause", paused: true, revision }, { type: "pause", paused: false, revision }]);
    }

    renderer.dispose();
  });
});

describe("CpuRenderer frame phases", () => {
  it("fills the cache grid level by level, shares its records with every worker, then shades", async() => {
    vi.stubGlobal("crossOriginIsolated", false);
    const renderer = create();
    const [worker] = FakeWorker.instances;
    const phases: string[] = [];
    renderer.on("progress", ({ phase, completed, total }) => phases.push(`${phase} ${completed}/${total}`));
    const done = renderer.render();
    await settle();
    const { revision } = worker.last("frame")!;
    // Two grid levels: two bands, then one finer band, before the single shading bucket.
    worker.reply({ type: "start", revision, width: 32, height: 32, order: [0], denoise: false, prepass: [2, 1] });
    expect(worker.last("render")).toMatchObject({ bucket: 0, phase: "cache" });
    worker.reply({ type: "cached", revision, job: 0, cells: new Float32Array([1, 2]) });
    expect(worker.last("render")).toMatchObject({ bucket: 1, phase: "cache" });
    worker.reply({ type: "cached", revision, job: 1, cells: new Float32Array([3]) });
    expect(worker.last("render")).toMatchObject({ bucket: 2, phase: "cache" });
    expect(worker.last("records")).toBeUndefined();
    worker.reply({ type: "cached", revision, job: 2, cells: new Float32Array([4]) });
    expect(Array.from(worker.last("records")!.cells)).toEqual([1, 2, 3, 4]);
    expect(worker.last("render")).toMatchObject({ bucket: 0, phase: "shade" });
    worker.reply({ type: "bucket", revision, bucket: { x: 0, y: 0, width: 32, height: 32, index: 0 },
      pixels: new Uint8ClampedArray(32 * 32 * 4) });
    await expect(done).resolves.toBe(renderer.image);
    // The last band hands over to shading directly, and the frame reports its end once.
    expect(phases).toEqual(["prepare 0/0", "cache 0/3", "cache 1/3", "cache 2/3", "shade 0/1", "shade 1/1"]);
  });

  it("hands the bucket accumulations to the denoiser and completes only when it answers", async() => {
    vi.stubGlobal("crossOriginIsolated", false);
    const renderer = create();
    const [worker] = FakeWorker.instances;
    const complete = vi.fn(), phases: string[] = [], painted: RenderPixels[] = [];
    renderer.on("complete", complete);
    renderer.on("pixels", pixels => painted.push(pixels));
    renderer.on("progress", ({ phase }) => phases.push(phase));
    const done = renderer.render();
    await settle();
    const { revision } = worker.last("frame")!;
    worker.reply({ type: "start", revision, width: 32, height: 32, order: [0, 1], denoise: true, prepass: [] });
    const pixels = new Uint8ClampedArray(16 * 32 * 4), first = new Float32Array(8), second = new Float32Array(8);
    worker.reply({ type: "bucket", revision, bucket: { x: 0, y: 0, width: 32, height: 16, index: 0 }, pixels, aov: first });
    expect(worker.last("denoise")).toBeUndefined();
    worker.reply({ type: "bucket", revision, bucket: { x: 0, y: 16, width: 32, height: 16, index: 1 }, pixels, aov: second });
    const denoise = worker.last("denoise")!;
    expect(denoise.slices.map(slice => slice.data)).toEqual([first, second]);
    // The accumulations are transferred to the worker, not copied.
    expect(worker.postMessage.mock.calls.at(-1)![1]).toEqual([first.buffer, second.buffer]);
    expect(phases.at(-1)).toBe("denoise");
    expect(complete).not.toHaveBeenCalled();
    expect(renderer.rendering).toBe(true);

    // The denoised image comes back in row bands, handed out at their offset.
    const rows = new Uint8ClampedArray(32 * 4 * 4).fill(7);
    worker.reply({ type: "pixels", revision, bucket: { x: 0, y: 0, width: 32, height: 32, index: -1 }, row: 8,
      pixels: rows });
    expect(painted.at(-1)).toEqual({ x: 0, y: 8, width: 32, height: 4, data: rows });
    expect(renderer.image!.data[8 * 32 * 4]).toBe(7);
    expect(renderer.image!.data[12 * 32 * 4]).toBe(0);
    worker.reply({ type: "denoised", revision });
    await expect(done).resolves.toBe(renderer.image);
    expect(complete).toHaveBeenCalledOnce();
  });

  it("allocates the image on start, hands out live rows of a bucket and reports which worker is on it", async() => {
    vi.stubGlobal("crossOriginIsolated", false);
    const renderer = create();
    const [worker] = FakeWorker.instances;
    const buckets: unknown[] = [], starts: unknown[] = [], painted: RenderPixels[] = [];
    renderer.on("bucket", event => buckets.push(event));
    renderer.on("start", event => starts.push(event));
    renderer.on("pixels", event => painted.push(event));
    expect(renderer.image).toBeNull();
    void renderer.render().catch(() => {});
    await settle();
    const { revision } = worker.last("frame")!;
    worker.reply({ type: "start", revision, width: 40, height: 20, order: [0], denoise: false, prepass: [] });
    expect(starts).toEqual([{ width: 40, height: 20 }]);
    const image = renderer.image!;
    expect([image.width, image.height, image.data.length]).toEqual([40, 20, 40 * 20 * 4]);
    const bucket = { x: 8, y: 4, width: 4, height: 4, index: 0 };
    worker.reply({ type: "active", revision, bucket });
    expect(buckets.at(-1)).toEqual({ worker: 0, bucket, width: 40, height: 20 });
    // One row of the bucket, resolved so far: it lands at the bucket's x and at the row's y.
    const row = new Uint8ClampedArray(4 * 4).fill(9);
    worker.reply({ type: "pixels", revision, bucket, row: 2, pixels: row });
    expect(painted.at(-1)).toEqual({ x: 8, y: 6, width: 4, height: 1, data: row });
    expect(Array.from(image.data.subarray((6 * 40 + 8) * 4, (6 * 40 + 12) * 4))).toEqual(Array(16).fill(9));
    expect(image.data[(6 * 40 + 12) * 4]).toBe(0);
    expect(image.data[(7 * 40 + 8) * 4]).toBe(0);
    const whole = new Uint8ClampedArray(4 * 4 * 4).fill(3);
    worker.reply({ type: "bucket", revision, bucket, pixels: whole });
    expect(painted.at(-1)).toEqual({ x: 8, y: 4, width: 4, height: 4, data: whole });
    expect(image.data[(6 * 40 + 8) * 4]).toBe(3);
    expect(buckets.at(-1)).toEqual({ worker: 0, bucket: null, width: 40, height: 20 });
    // The next frame gets its own buffer: the finished image stays intact for whoever kept it.
    void renderer.render().catch(() => {});
    await settle();
    worker.reply({ type: "start", revision: revision + 1, width: 40, height: 20, order: [0], denoise: false, prepass: [] });
    expect(renderer.image).not.toBe(image);
    expect(image.data[(6 * 40 + 8) * 4]).toBe(3);
    renderer.dispose();
  });

  it("forwards the owner's preparation to the other workers, which then take buckets too", async() => {
    vi.stubGlobal("crossOriginIsolated", true);
    const renderer = create({ workers: 3 });
    const [owner, ...readers] = FakeWorker.instances;
    void renderer.render().catch(() => {});
    await settle();
    const { revision } = owner.last("frame")!;
    const frame = { width: 32 } as never;
    owner.reply({ type: "prepared", revision, frame });
    expect(owner.last("adopt")).toBeUndefined();
    for (const reader of readers)
      expect(reader.last("adopt")).toEqual({ type: "adopt", revision, frame });

    owner.reply({ type: "start", revision, width: 32, height: 32, order: [4, 5, 6], denoise: false, prepass: [] });
    readers[0].reply({ type: "start", revision, width: 32, height: 32, order: [4, 5, 6], denoise: false, prepass: [] });
    expect(owner.last("render")!.bucket).toBe(4);
    expect(readers[0].last("render")!.bucket).toBe(5);
    expect(readers[1].last("render")).toBeUndefined();
    renderer.dispose();
  });
});

describe("CpuRenderer inputs", () => {
  it("merges settings group by group and sends them to the workers in the engine's flat form", async() => {
    vi.stubGlobal("crossOriginIsolated", false);

    const renderer = create({ settings: { depthOfField: { enabled: true }, background: { mode: "color" } } });

    const [worker] = FakeWorker.instances;
    renderer.setSettings({ depthOfField: { aperture: 1.4 }, toneMapping: "agx" });
    expect(renderer.settings.depthOfField).toEqual({ enabled: true, focusDistance: 10, aperture: 1.4, blades: 0 });
    expect(renderer.settings.background).toEqual({ mode: "color", color: "#1a1a1a" });
    void renderer.render().catch(() => {});
    await settle();
    expect(worker.last("frame")!.settings).toMatchObject({
      dofEnabled: true, dofAperture: 1.4, dofFocusDistance: 10, tonemapping: "agx", backgroundMode: "color",
      maxSamples: 4, globalIllumination: false,
    });
    renderer.dispose();
  });

  it("reloads the environment only when its image changes, not its intensity or rotation", async() => {
    vi.stubGlobal("crossOriginIsolated", false);
    const renderer = create({ environment: { kind: "hdri", url: "a.hdr" } });
    const [worker] = FakeWorker.instances;

    const frame = async() => {
      void renderer.render().catch(() => {});
      await settle();

      return worker.last("frame")!;
    };

    expect((await frame()).environment).toEqual({ kind: "hdri", url: "a.hdr" });
    renderer.setEnvironment({ kind: "hdri", url: "a.hdr", intensity: 0.5, rotation: 90 });
    const dimmed = await frame();
    expect(dimmed.environment).toBeUndefined();
    expect(dimmed.settings).toMatchObject({ environmentIntensity: 0.5, environmentRotation: 90 });
    renderer.setEnvironment({ kind: "hdri", url: "b.hdr", intensity: 0.5, rotation: 90 });
    expect((await frame()).environment).toEqual({ kind: "hdri", url: "b.hdr" });
    renderer.dispose();
  });

  it("serializes a Three scene and reads a Three camera as it is at each render()", async() => {
    vi.stubGlobal("crossOriginIsolated", false);
    const scene = new THREE.Scene();
    scene.add(new THREE.Mesh(new THREE.BoxGeometry(), new THREE.MeshStandardMaterial()));
    const camera = new THREE.PerspectiveCamera(35, 1, 0.5, 20);
    camera.position.set(1, 2, 3);
    const renderer = create({ scene, camera });
    const [worker] = FakeWorker.instances;
    void renderer.render().catch(() => {});
    await settle();
    const frame = worker.last("frame")!;
    expect(frame.scene!.meshes).toHaveLength(1);
    expect(frame.camera).toMatchObject({ matrixWorld: camera.matrixWorld.toArray(), fov: 35, near: 0.5, far: 20,
      type: "perspective", aspect: 1 });
    // Without shared memory the geometry buffers are transferred along with the frame.
    expect(worker.postMessage.mock.calls.at(-1)![1]!.length).toBeGreaterThan(0);

    // A resize changes the aspect after the camera was set: the next frame projects with it, not a stale matrix.
    camera.aspect = 2;
    camera.position.set(4, 0, 0);
    void renderer.render().catch(() => {});
    await settle();
    const resized = worker.last("frame")!.camera;
    expect(resized.aspect).toBe(2);
    expect(resized.projectionMatrix).toEqual(camera.projectionMatrix.toArray());
    expect(resized.matrixWorld.slice(12, 15)).toEqual([4, 0, 0]);
    renderer.dispose();
  });

  it("keeps a camera snapshot fixed and reads the camera when render() is called", async() => {
    vi.stubGlobal("crossOriginIsolated", false);
    const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 10);
    const snapshot = serializeCamera(camera);
    const renderer = create({ camera: snapshot });
    const [worker] = FakeWorker.instances;
    camera.position.set(9, 9, 9);
    void renderer.render().catch(() => {});
    await settle();
    expect(worker.last("frame")!.camera).toBe(snapshot);

    // Moving a Three camera after render() was called, while the scene serializes, does not change that frame.
    renderer.setCamera(camera);
    void renderer.render().catch(() => {});
    camera.position.set(0, 0, 0);
    await settle();
    expect(worker.last("frame")!.camera.matrixWorld.slice(12, 15)).toEqual([9, 9, 9]);
    renderer.dispose();
  });

  it("renders the scene set last when another one is set while the first is still serializing", async() => {
    vi.stubGlobal("crossOriginIsolated", false);
    const first = new THREE.Scene();
    first.add(new THREE.Mesh(new THREE.BoxGeometry(), new THREE.MeshBasicMaterial()));
    const renderer = create({ scene: first });
    const [worker] = FakeWorker.instances;
    const done = renderer.render();
    renderer.setScene(emptyScene());
    await settle();
    await settle();
    expect(worker.last("frame")!.scene!.meshes).toHaveLength(0);
    renderer.dispose();
    await expect(done).rejects.toSatisfy(isAbortError);
  });

  it("reports a scene that cannot be serialized and recovers with the next scene", async() => {
    vi.stubGlobal("crossOriginIsolated", false);
    const scene = new THREE.Scene(), material = new THREE.MeshBasicMaterial();
    // Stencil operations have no CPU equivalent: serialization refuses them.
    material.stencilWrite = true;
    scene.add(new THREE.Mesh(new THREE.PlaneGeometry(), material));
    const renderer = create({ scene });
    const [worker] = FakeWorker.instances;
    const errors: string[] = [];
    renderer.on("error", error => errors.push(error.message));
    await expect(renderer.render()).rejects.toThrow("stencil operations are unsupported");
    expect(errors).toHaveLength(1);
    expect(renderer.rendering).toBe(false);
    expect(worker.last("frame")).toBeUndefined();
    renderer.setScene(emptyScene());
    void renderer.render().catch(() => {});
    await settle();
    expect(worker.last("frame")!.scene).toEqual(emptyScene().scene);
    renderer.dispose();
  });

  it("sends the size and pixel ratio it was given and the ones set afterwards", async() => {
    vi.stubGlobal("crossOriginIsolated", false);
    const renderer = create({ width: 300, height: 150, pixelRatio: 1.5 });
    const [worker] = FakeWorker.instances;
    void renderer.render().catch(() => {});
    await settle();
    expect(worker.last("frame")).toMatchObject({ width: 300, height: 150, pixelRatio: 1.5 });
    // The size is whole CSS pixels; the pixel ratio stays unless given.
    renderer.setSize(10.7, 5.2);
    void renderer.render().catch(() => {});
    await settle();
    expect(worker.last("frame")).toMatchObject({ width: 10, height: 5, pixelRatio: 1.5 });
    renderer.setSize(0, -4, 3);
    void renderer.render().catch(() => {});
    await settle();
    expect(worker.last("frame")).toMatchObject({ width: 1, height: 1, pixelRatio: 3 });
    renderer.setPixelRatio(2);
    void renderer.render().catch(() => {});
    await settle();
    expect(worker.last("frame")).toMatchObject({ width: 1, height: 1, pixelRatio: 2 });
    expect(() => renderer.setPixelRatio(0)).toThrow("pixelRatio");
    expect(() => create({ pixelRatio: Number.NaN })).toThrow("pixelRatio");
    renderer.dispose();
  });

  it("fails the frame when a worker cannot load or a message cannot be decoded", async() => {
    vi.stubGlobal("crossOriginIsolated", false);
    const renderer = create();

    const [worker] = FakeWorker.instances as (FakeWorker & { onerror: (event: { message: string }) => void;
      onmessageerror: () => void })[];

    const loading = renderer.render();
    await settle();
    worker.onerror({ message: "" });
    await expect(loading).rejects.toThrow("Unable to load the CPU render worker.");
    const decoding = renderer.render();
    await settle();
    worker.onmessageerror();
    await expect(decoding).rejects.toThrow("Unable to decode a render worker message.");
  });

  it("ignores worker output after dispose", async() => {
    vi.stubGlobal("crossOriginIsolated", false);
    const renderer = create();
    const [worker] = FakeWorker.instances;
    const listener = vi.fn();
    renderer.on("start", listener);
    renderer.on("error", listener);
    const done = renderer.render();
    await settle();
    const { revision } = worker.last("frame")!;
    renderer.dispose();
    await expect(done).rejects.toSatisfy(isAbortError);
    worker.reply({ type: "start", revision, width: 32, height: 32, order: [0], denoise: false, prepass: [] });
    worker.reply({ type: "error", revision, message: "late" });
    expect(listener).not.toHaveBeenCalled();
    // Dispose is final and idempotent.
    renderer.dispose();
    expect(worker.terminate).toHaveBeenCalledOnce();
  });
});
