import { PerspectiveCamera, type OrthographicCamera, type Scene } from "three";
import {
  DEFAULT_RENDER_ENVIRONMENT,
  DEFAULT_RENDER_SETTINGS,
  environmentSource,
  mergeSettings,
  toFrameSettings,
  type CameraState,
  type RenderEnvironment,
  type RenderSettings,
  type RenderSettingsInput,
} from "./settings.js";
import type { BucketAovSlice, FramePhase, WorkerRequest, WorkerResponse } from "./protocol.js";
import { serializeScene, type SerializedSceneResult } from "./sceneSerialization.js";
import { serializeCamera } from "./camera.js";
import type { Bucket } from "./rasterizer.js";
import { shareable } from "./sharedBuffers.js";
import { BucketScheduler } from "./bucketScheduler.js";
import { EventEmitter } from "./events.js";

/** Where a frame is: preparing the scene, filling the irradiance cache grid, shading buckets or denoising. */
export type RenderPhase = "prepare" | FramePhase | "denoise";

/** A Three camera, read at every `render()`, or a fixed snapshot from `serializeCamera`. */
export type RenderCamera = PerspectiveCamera | OrthographicCamera | CameraState;

/** A Three scene, serialized when set, or the result of `serializeScene`. */
export type RenderScene = Scene | SerializedSceneResult;

/** RGBA pixels of a region of the image: `data` holds `width * height * 4` bytes, row by row. */
export interface RenderPixels {
  x: number;
  y: number;
  width: number;
  height: number;
  data: Uint8ClampedArray<ArrayBuffer>;
}

/** The whole image in RGBA bytes, straight sRGB, `width * height * 4` long. */
export interface RenderImage {
  width: number;
  height: number;
  data: Uint8ClampedArray<ArrayBuffer>;
}

export interface CpuRendererEvents {
  /** The workers prepared the frame and are shading; `image` has the render size and is empty. */
  start: { width: number; height: number };
  /** Finished pixels of a region, as they arrive: paint them at (`x`, `y`). `image` already holds them. */
  pixels: RenderPixels;
  /** Work done out of `total`: bands of the cache grid in the cache phase, buckets otherwise. */
  progress: { completed: number; total: number; phase: RenderPhase };
  /** The bucket a worker is on, or `null` when it is idle; `width` and `height` are the render size. */
  bucket: { worker: number; bucket: Bucket | null; width: number; height: number };
  /** `image` is finished. */
  complete: void;
  /** The frame failed; `render()` rejects with the same error. */
  error: Error;
}

export interface CpuRendererOptions {
  scene?: RenderScene;
  /** Defaults to a perspective camera at (5, 5, 5) looking at the origin. */
  camera?: RenderCamera;
  /** Merged over `DEFAULT_RENDER_SETTINGS`, group by group. */
  settings?: RenderSettingsInput;
  /** Defaults to `DEFAULT_RENDER_ENVIRONMENT`. */
  environment?: RenderEnvironment;
  /** Image size in CSS pixels; the render size also applies `pixelRatio` and `settings.renderScale`. */
  width: number;
  height: number;
  /** Defaults to 1. Pass `window.devicePixelRatio` for a sharp image on a high-density screen. */
  pixelRatio?: number;
  /** Pool size. Defaults to `defaultWorkerCount()`; capped at `MAX_RENDER_WORKERS` and at one without shared memory. */
  workers?: number;
  /**
   * Replace it when the bundler cannot follow `new URL("./worker.js", import.meta.url)`, or outside the browser:
   * `@tinypoly/cpu-renderer/node` exports `createNodeWorker` for `worker_threads`.
   */
  createWorker?: WorkerFactory;
}

/**
 * What the renderer needs from a worker: the shape of a Web `Worker`. Any transport that carries structured clones
 * and transfer lists fits, such as a `worker_threads` thread behind an adapter.
 */
export interface RenderWorker {
  postMessage(message: WorkerRequest, transfer?: Transferable[]): void;
  terminate(): void;
  onmessage: ((event: MessageEvent<WorkerResponse>) => void) | null;
  onerror: ((event: ErrorEvent) => void) | null;
  onmessageerror: ((event: MessageEvent) => void) | null;
}

/** Creates one render worker. */
export type WorkerFactory = () => RenderWorker;

/** Loads `worker.js` next to this module, a pattern Vite, webpack 5 and Parcel resolve in dependencies. */
export function createDefaultWorker(): RenderWorker {
  if (typeof Worker === "undefined")
    throw new Error("Web Workers are unavailable here. In Node, pass `createWorker: createNodeWorker` from "
      + "@tinypoly/cpu-renderer/node.");

  return new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
}

/** Manual upper limit; automatic mode uses a smaller pool to bound per-worker scratch memory. */
export const MAX_RENDER_WORKERS = 32;

/** One core stays free for the UI and for scene serialization. `cores` defaults to `navigator.hardwareConcurrency`. */
export function defaultWorkerCount(cores = detectCores()): number {
  return shareable() ? Math.max(1, Math.min(8, Math.floor(cores) - 1)) : 1;
}

/** Browsers and Node 21+ report it on `navigator`; elsewhere, four is a reasonable guess. */
function detectCores(): number {
  return typeof navigator !== "undefined" && navigator.hardwareConcurrency ? navigator.hardwareConcurrency : 4;
}

/** Without shared memory, multiple readers would each clone the whole prepared scene. */
export function effectiveWorkerCount(requested: number): number {
  return shareable() ? Math.max(1, Math.min(MAX_RENDER_WORKERS, Math.floor(requested) || 1)) : 1;
}

/** `render()` rejects with this when a newer frame, or `dispose()`, replaced the frame it started. */
export function isAbortError(error: unknown): boolean {
  return error instanceof Error && error.name === "AbortError";
}

function abortError(message: string): Error {
  const error = new Error(message);
  error.name = "AbortError";

  return error;
}

interface Size {
  width: number;
  height: number;
  pixelRatio: number;
}

interface PendingFrame {
  resolve: (image: RenderImage) => void;
  reject: (reason: Error) => void;
}

type RequestPayload<T = WorkerRequest> = T extends WorkerRequest ? Omit<T, "revision"> : never;

/**
 * Renders a Three.js scene on the CPU, in a pool of workers. It has no output of its own: `render()` resolves with
 * the finished image, and on the way there finished pixels arrive through the `pixels` event and accumulate in
 * `image`. `attachCanvas` paints them on a Canvas 2D.
 *
 * Setters only record the inputs; `render()` starts a frame with the current ones. Progress arrives through events.
 */
export class CpuRenderer extends EventEmitter<CpuRendererEvents> {
  private workers: RenderWorker[];
  private output: RenderImage | null = null;
  private revision = 0;
  private ticket = 0;
  private disposed = false;
  private isPaused = false;
  private scheduler = new BucketScheduler();
  private started = false;
  private frame: PendingFrame | null = null;
  // The render() call in flight, from the call itself until its frame settles.
  private active: number | null = null;
  // The frame ends with the denoiser on the first worker; until it responds, it is still rendering.
  private denoise = false;
  private denoising = false;
  private slices: BucketAovSlice[] = [];
  // Irradiance cache phase: the whole-image grid, level by level, before any shading.
  private phase: FramePhase = "shade";
  private order: number[] = [];
  private readyWorkers = new Set<number>();
  private prepass: number[] = [];
  private level = 0;
  private cacheCompleted = 0;
  private cells: Float32Array<ArrayBuffer>[] = [];

  // Inputs of the next frame. Scene and environment are sent only when they changed.
  private current: RenderSettings;
  // A Three camera is kept by reference and read at every frame, as WebGLRenderer does; a snapshot stays fixed.
  private camera: RenderCamera | null = null;
  private environment: RenderEnvironment;
  // The loadable part of the environment the workers last received; intensity and rotation travel in the settings.
  private sentEnvironment: string | null = null;
  private nextScene: Promise<SerializedSceneResult> | null = null;
  private size: Size;

  constructor(options: CpuRendererOptions) {
    super();
    this.current = mergeSettings(DEFAULT_RENDER_SETTINGS, options.settings);
    this.environment = options.environment ?? DEFAULT_RENDER_ENVIRONMENT;
    this.size = imageSize(options.width, options.height, options.pixelRatio ?? 1);
    if (options.camera)
      this.setCamera(options.camera);
    this.setScene(options.scene ?? { scene: { meshes: [], lights: [], textures: [] }, transfer: [] });

    const createWorker = options.createWorker ?? createDefaultWorker;

    this.workers = Array.from({ length: effectiveWorkerCount(options.workers ?? defaultWorkerCount()) }, (_, index) => {
      const worker = createWorker();
      worker.onmessage = event => this.handle(index, event.data);
      worker.onerror = event => this.fail(new Error(event.message || "Unable to load the CPU render worker."));
      worker.onmessageerror = () => this.fail(new Error("Unable to decode a render worker message."));

      return worker;
    });
  }

  /** Number of workers actually running. */
  get workerCount() {
    return this.workers.length;
  }

  /** The settings the next frame renders with. */
  get settings(): Readonly<RenderSettings> {
    return this.current;
  }

  /** A frame is in flight: `render()` was called and has not settled. */
  get rendering() {
    return this.active !== null;
  }

  get paused() {
    return this.isPaused;
  }

  /**
   * The image of the frame in flight, filled as buckets finish, or of the last one; `null` before the first frame
   * starts. `render()` resolves with the finished one. Each frame allocates a new buffer, so a reference taken at
   * `complete` stays as it is.
   */
  get image(): RenderImage | null {
    return this.output;
  }

  /** Takes a Three scene, serialized in the background, or the result of `serializeScene`. */
  setScene(scene: RenderScene) {
    this.nextScene = isThreeScene(scene) ? serializeScene(scene) : Promise.resolve(scene);
  }

  /**
   * A Three camera is read at every `render()`, so moving it or changing its aspect only needs a new frame.
   * A `CameraState` snapshot is used as it is.
   */
  setCamera(camera: RenderCamera) {
    this.camera = camera;
  }

  /** Merges the given fields into the current settings, group by group. */
  setSettings(settings: RenderSettingsInput) {
    this.current = mergeSettings(this.current, settings);
  }

  setEnvironment(environment: RenderEnvironment) {
    this.environment = environment;
  }

  /** Image size in CSS pixels for the next frame; `pixelRatio` stays as it is when omitted. */
  setSize(width: number, height: number, pixelRatio = this.size.pixelRatio) {
    this.size = imageSize(width, height, pixelRatio);
  }

  setPixelRatio(pixelRatio: number) {
    this.size = imageSize(this.size.width, this.size.height, pixelRatio);
  }

  /**
   * Starts a frame with the current scene, camera, settings, environment and size, discarding any frame in
   * progress. Resolves with the finished image. Rejects with an `AbortError` when a later `render()` or `dispose()`
   * replaces the frame, and with the worker's error when the frame fails.
   */
  render(): Promise<RenderImage> {
    const ticket = ++this.ticket;

    // The camera as it is at this call, even if it moves while the scene is still serializing.
    const camera = !this.camera ? defaultCamera()
      : isThreeCamera(this.camera) ? serializeCamera(this.camera) : this.camera;

    if (!this.disposed)
      this.active = ticket;

    return this.begin(ticket, camera).catch((error: unknown) => {
      if (this.active === ticket)
        this.active = null;
      throw error;
    });
  }

  private async begin(ticket: number, camera: CameraState): Promise<RenderImage> {
    if (this.disposed)
      throw new Error("The renderer is disposed.");
    let scene: SerializedSceneResult | null = null;

    // Serialization is asynchronous; a scene set meanwhile replaces the one being awaited.
    while (this.nextScene) {
      const pending = this.nextScene;
      let result: SerializedSceneResult;

      try {
        result = await pending;
      } catch(error) {
        const failure = error instanceof Error ? error : new Error(String(error));
        if (this.nextScene === pending)
          this.nextScene = null;
        if (ticket === this.ticket && !this.disposed)
          this.emit("error", failure);
        throw failure;
      }

      if (this.disposed)
        throw new Error("The renderer is disposed.");
      if (ticket !== this.ticket)
        throw abortError("A newer render() replaced this frame.");

      if (this.nextScene === pending) {
        scene = result;
        this.nextScene = null;
      }
    }

    const size = this.size;
    // Only a different image or gradient is loaded again; intensity and rotation apply at render time.
    const source = environmentSource(this.environment), key = JSON.stringify(source);
    const environment = key === this.sentEnvironment ? undefined : source;
    this.sentEnvironment = key;
    this.abort("A newer render() replaced this frame.");

    const settled = new Promise<RenderImage>((resolve, reject) => {
      this.frame = { resolve, reject };
    });

    this.active = ticket;

    this.post({
      type: "frame",
      publishPrepared: this.workers.length > 1,
      ...size,
      settings: toFrameSettings(this.current, this.environment),
      camera,
      scene: scene?.scene,
      environment,
    }, scene?.transfer ?? []);

    return settled;
  }

  /** Stops every worker where it is; `resume()` continues the same frame. */
  pause() {
    if (this.isPaused)
      return;
    this.isPaused = true;
    this.broadcast({ type: "pause", paused: true });
  }

  resume() {
    if (!this.isPaused)
      return;
    this.isPaused = false;
    this.broadcast({ type: "pause", paused: false });
  }

  /** Terminates the workers. A frame in flight rejects with an `AbortError`. */
  dispose() {
    if (this.disposed)
      return;
    this.broadcast({ type: "dispose" });
    this.disposed = true;
    this.abort("The renderer was disposed.");
    for (const worker of this.workers)
      worker.terminate();
  }

  /** Render size of the frame that started last, for the bucket events. */
  private get rendered() {
    return { width: this.output?.width ?? 0, height: this.output?.height ?? 0 };
  }

  private clearActiveBuckets() {
    for (let index = 0; index < this.workers.length; index++)
      this.emit("bucket", { worker: index, bucket: null, ...this.rendered });
  }

  /** Copies finished rows into `image` and hands them out; rows outside the image are dropped. */
  private paint(x: number, y: number, width: number, data: Uint8ClampedArray<ArrayBuffer>) {
    const image = this.output;
    if (!image || width < 1)
      return;
    const height = Math.floor(data.length / 4 / width);
    const columns = Math.min(width, image.width - x), rows = Math.min(height, image.height - y);
    if (x < 0 || y < 0 || columns < 1 || rows < 1)
      return;

    for (let row = 0; row < rows; row++)
      image.data.set(data.subarray(row * width * 4, row * width * 4 + columns * 4), ((y + row) * image.width + x) * 4);

    this.emit("pixels", { x, y, width, height, data });
  }

  /** Settles the frame in flight, if any, with an `AbortError`. */
  private abort(reason: string) {
    const frame = this.settle();
    frame?.reject(abortError(reason));
  }

  private fail(error: Error) {
    const frame = this.settle();
    this.clearActiveBuckets();
    this.emit("error", error);
    frame?.reject(error);
  }

  private finish() {
    const frame = this.settle();
    this.emit("complete");
    if (frame && this.output)
      frame.resolve(this.output);
    else
      frame?.reject(new Error("The frame finished before it started."));
  }

  /** Takes the frame in flight, if any, so it can be settled after the events go out. */
  private settle(): PendingFrame | null {
    const frame = this.frame;
    this.frame = null;
    if (frame)
      this.active = null;

    return frame;
  }

  private emitProgress() {
    if (!this.frame)
      return;
    const cache = this.phase === "cache";
    this.emit("progress", {
      completed: cache ? this.cacheCompleted + this.scheduler.completed : this.scheduler.completed,
      total: cache ? this.prepass.reduce((a, b) => a + b, 0) : this.scheduler.total,
      phase: !this.started ? "prepare" : this.denoising ? "denoise" : this.phase,
    });
  }

  /** Job indices of one cache-phase level: levels are contiguous in the worker's list. */
  private levelJobs(level: number): number[] {
    const offset = this.prepass.slice(0, level).reduce((a, b) => a + b, 0);

    return Array.from({ length: this.prepass[level] }, (_, i) => offset + i);
  }

  /** Restarts the scheduler with another list and hands the first item to workers that were already ready. */
  private restart(order: number[]) {
    this.scheduler.reset(order);

    for (const ready of this.readyWorkers) {
      const first = this.scheduler.markReady(ready);
      if (first !== null)
        this.dispatch(ready, first);
    }
  }

  /** Pause and dispose go to every worker, without opening a revision. */
  private broadcast(message: RequestPayload) {
    if (this.disposed)
      return;
    for (const worker of this.workers)
      worker.postMessage({ ...message, revision: this.revision });
  }

  /**
   * A frame opens a new revision and goes only to the first worker, which prepares; the others drop what they
   * were doing and wait for the preparation to adopt it.
   */
  private post(message: RequestPayload, transfer: Transferable[]) {
    this.revision++;
    this.started = false;
    this.denoising = false;
    this.slices = [];
    this.phase = "shade";
    this.order = [];
    this.readyWorkers.clear();
    this.prepass = [];
    this.level = 0;
    this.cacheCompleted = 0;
    this.cells = [];
    this.scheduler.reset([]);
    this.clearActiveBuckets();
    this.emitProgress();
    const [owner, ...readers] = this.workers;
    owner.postMessage({ ...message, revision: this.revision }, transfer);
    for (const reader of readers)
      reader.postMessage({ type: "reset", revision: this.revision } satisfies WorkerRequest);
  }

  private dispatch(worker: number, bucket: number) {
    this.workers[worker].postMessage(
      { type: "render", revision: this.revision, bucket, phase: this.phase } satisfies WorkerRequest);
  }

  /** A worker returned a bucket of the current phase; when the phase ends, the next one starts. */
  private completeBucket(worker: number, bucket: number) {
    const next = this.scheduler.complete(worker, bucket);
    if (next !== null)
      this.dispatch(worker, next);
    else
      this.emit("bucket", { worker, bucket: null, ...this.rendered });
    if (this.scheduler.total === 0 || this.scheduler.busy)
      return;

    if (this.phase === "cache") {
      this.cacheCompleted += this.scheduler.completed;
      this.level++;

      if (this.level < this.prepass.length) {
        // Next, finer grid level, only where the previous one does not reach.
        this.restart(this.levelJobs(this.level));

        return;
      }

      // Grid done over the whole image: without shared memory, every worker receives what the others wrote.
      if (this.cells.length) {
        const total = this.cells.reduce((a, c) => a + c.length, 0), cells = new Float32Array(total);
        let cursor = 0;

        for (const chunk of this.cells) {
          cells.set(chunk, cursor);
          cursor += chunk.length;
        }

        this.cells = [];
        for (const worker of this.workers)
          worker.postMessage({ type: "records", revision: this.revision, cells: Float32Array.from(cells) } satisfies WorkerRequest);
      }

      this.phase = "shade";
      this.restart(this.order);
    } else if (this.denoise) {
      // Last bucket painted: the first worker denoises the indirect light and resends the image in bands.
      this.denoising = true;
      const slices = this.slices;
      this.slices = [];
      this.workers[0].postMessage({ type: "denoise", revision: this.revision, slices } satisfies WorkerRequest,
        slices.map(slice => slice.data.buffer));
    } else {
      this.emitProgress();
      this.finish();
    }
  }

  private handle(worker: number, message: WorkerResponse) {
    if (this.disposed || message.revision !== this.revision)
      return;

    switch (message.type) {
      case "prepared":
        // With shared memory the geometry is not copied: each worker only receives the references.
        for (let index = 1; index < this.workers.length; index++)
          this.workers[index].postMessage(
            { type: "adopt", revision: this.revision, frame: message.frame } satisfies WorkerRequest);
        break;

      case "start": {
        if (!this.started) {
          this.started = true;

          this.output = {
            width: message.width,
            height: message.height,
            data: new Uint8ClampedArray(message.width * message.height * 4),
          };

          this.denoise = message.denoise;
          this.order = message.order;
          this.prepass = message.prepass;
          this.level = 0;
          this.cacheCompleted = 0;
          this.phase = this.prepass.length ? "cache" : "shade";
          this.scheduler.reset(this.phase === "cache" ? this.levelJobs(0) : this.order);
          this.emit("start", { width: message.width, height: message.height });
        }

        this.readyWorkers.add(worker);
        const bucket = this.scheduler.markReady(worker);
        if (bucket !== null)
          this.dispatch(worker, bucket);
        this.emitProgress();
        break;
      }

      case "bucket": {
        this.paint(message.bucket.x, message.bucket.y, message.bucket.width, message.pixels);
        if (message.aov)
          this.slices.push({ bucket: message.bucket, data: message.aov });
        this.completeBucket(worker, message.bucket.index);
        this.emitProgress();
        break;
      }

      case "cached":
        if (message.cells?.length)
          this.cells.push(message.cells);
        this.completeBucket(worker, message.job);
        this.emitProgress();
        break;
      case "denoised":
        this.denoising = false;
        this.emitProgress();
        this.finish();
        break;
      case "pixels":
        // Only the received row band; the full bucket still arrives later in "bucket".
        this.paint(message.bucket.x, message.bucket.y + message.row, message.bucket.width, message.pixels);
        break;
      case "active":
        this.emit("bucket", { worker, bucket: message.bucket, ...this.rendered });
        break;
      case "error":
        this.fail(new Error(message.message));
        break;
    }
  }
}

function isThreeScene(scene: RenderScene): scene is Scene {
  return "isScene" in scene && scene.isScene === true;
}

function isThreeCamera(camera: RenderCamera): camera is PerspectiveCamera | OrthographicCamera {
  return "isCamera" in camera && camera.isCamera === true;
}

function imageSize(width: number, height: number, pixelRatio: number): Size {
  if (!Number.isFinite(pixelRatio) || pixelRatio <= 0)
    throw new Error("CPU renderer: pixelRatio must be a positive number.");

  return { width: Math.max(1, Math.floor(width)), height: Math.max(1, Math.floor(height)), pixelRatio };
}

/** A perspective camera at (5, 5, 5) looking at the origin, for a frame rendered before any camera is set. */
function defaultCamera(): CameraState {
  const camera = new PerspectiveCamera(50, 1, 0.1, 1000);
  camera.position.set(5, 5, 5);
  camera.lookAt(0, 0, 0);

  return serializeCamera(camera);
}
