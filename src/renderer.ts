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

export interface CpuRendererEvents {
  /** The workers prepared the frame and are shading; the canvas has the render size. */
  start: { width: number; height: number };
  /** Work done out of `total`: bands of the cache grid in the cache phase, buckets otherwise. */
  progress: { completed: number; total: number; phase: RenderPhase };
  /** The bucket a worker is on, or `null` when it is idle; `width` and `height` are the render size. */
  bucket: { worker: number; bucket: Bucket | null; width: number; height: number };
  /** The image on the canvas is finished. */
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
  /** Image size in CSS pixels. Without it, each `render()` measures the canvas' CSS size. */
  width?: number;
  height?: number;
  /** Defaults to `window.devicePixelRatio`. */
  pixelRatio?: number;
  /** Pool size. Defaults to `defaultWorkerCount()`; capped at `MAX_RENDER_WORKERS` and at one without shared memory. */
  workers?: number;
  /** Replace it when the bundler cannot follow `new URL("./worker.js", import.meta.url)`. */
  createWorker?: WorkerFactory;
}

/** Creates one render worker. */
export type WorkerFactory = () => Worker;

/** Loads `worker.js` next to this module, a pattern Vite, webpack 5 and Parcel resolve in dependencies. */
export function createDefaultWorker(): Worker {
  return new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
}

/** Manual upper limit; automatic mode uses a smaller pool to bound per-worker scratch memory. */
export const MAX_RENDER_WORKERS = 32;

/** One core stays free for the UI and for scene serialization. */
export function defaultWorkerCount(): number {
  const cores = typeof navigator !== "undefined" && navigator.hardwareConcurrency ? navigator.hardwareConcurrency : 4;

  return shareable() ? Math.max(1, Math.min(8, cores - 1)) : 1;
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
  resolve: () => void;
  reject: (reason: Error) => void;
}

type RequestPayload<T = WorkerRequest> = T extends WorkerRequest ? Omit<T, "revision"> : never;

/**
 * Renders a Three.js scene on the CPU, in a pool of Web Workers, into a 2D canvas.
 *
 * Setters only record the inputs; `render()` starts a frame with the current ones and resolves when the canvas
 * holds the finished image. Progress arrives through events.
 */
export class CpuRenderer extends EventEmitter<CpuRendererEvents> {
  private workers: Worker[];
  private context: CanvasRenderingContext2D;
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
  private size: Size | null = null;
  private pixelRatio: number | undefined;

  constructor(readonly canvas: HTMLCanvasElement, options: CpuRendererOptions = {}) {
    super();
    const context = canvas.getContext("2d");
    if (!context)
      throw new Error("Canvas 2D is unavailable.");
    this.context = context;
    this.current = mergeSettings(DEFAULT_RENDER_SETTINGS, options.settings);
    this.environment = options.environment ?? DEFAULT_RENDER_ENVIRONMENT;
    this.pixelRatio = options.pixelRatio;
    if (options.width !== undefined && options.height !== undefined)
      this.setSize(options.width, options.height, options.pixelRatio);
    if (options.camera)
      this.setCamera(options.camera);
    this.setScene(options.scene ?? { scene: { meshes: [], lights: [], textures: [] }, transfer: [] });

    const createWorker = options.createWorker ?? createDefaultWorker;

    this.workers = Array.from({ length: effectiveWorkerCount(options.workers ?? defaultWorkerCount()) }, (_, index) => {
      const worker = createWorker();
      worker.onmessage = (event: MessageEvent<WorkerResponse>) => this.handle(index, event.data);
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

  /** Fixes the image size in CSS pixels. Without it, each frame measures the canvas' CSS size. */
  setSize(width: number, height: number, pixelRatio = this.pixelRatio ?? defaultPixelRatio()) {
    this.size = { width: Math.max(1, Math.floor(width)), height: Math.max(1, Math.floor(height)), pixelRatio };
  }

  /**
   * Starts a frame with the current scene, camera, settings, environment and size, discarding any frame in
   * progress. Resolves when the canvas holds the finished image. Rejects with an `AbortError` when a later
   * `render()` or `dispose()` replaces the frame, and with the worker's error when the frame fails.
   */
  render(): Promise<void> {
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

  private async begin(ticket: number, camera: CameraState): Promise<void> {
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

    const size = this.size ?? this.measure();
    // Only a different image or gradient is loaded again; intensity and rotation apply at render time.
    const source = environmentSource(this.environment), key = JSON.stringify(source);
    const environment = key === this.sentEnvironment ? undefined : source;
    this.sentEnvironment = key;
    this.abort("A newer render() replaced this frame.");

    const settled = new Promise<void>((resolve, reject) => {
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

  private measure(): Size {
    return {
      width: Math.max(1, this.canvas.clientWidth || this.canvas.width),
      height: Math.max(1, this.canvas.clientHeight || this.canvas.height),
      pixelRatio: this.pixelRatio ?? defaultPixelRatio(),
    };
  }

  private clearActiveBuckets() {
    for (let index = 0; index < this.workers.length; index++)
      this.emit("bucket", { worker: index, bucket: null, width: this.canvas.width, height: this.canvas.height });
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
    frame?.resolve();
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
      this.emit("bucket", { worker, bucket: null, width: this.canvas.width, height: this.canvas.height });
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
          this.canvas.width = message.width;
          this.canvas.height = message.height;
          this.context.clearRect(0, 0, message.width, message.height);
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
        this.context.putImageData(
          new ImageData(message.pixels, message.bucket.width, message.bucket.height),
          message.bucket.x,
          message.bucket.y,
        );
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
        // Paints only the received row band; the full bucket still arrives later in "bucket".
        this.context.putImageData(
          new ImageData(message.pixels, message.bucket.width, message.pixels.length / 4 / message.bucket.width),
          message.bucket.x,
          message.bucket.y + message.row,
        );
        break;
      case "active":
        this.emit("bucket", { worker, bucket: message.bucket, width: this.canvas.width, height: this.canvas.height });
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

function defaultPixelRatio(): number {
  return typeof window !== "undefined" && window.devicePixelRatio ? window.devicePixelRatio : 1;
}

/** A perspective camera at (5, 5, 5) looking at the origin, for a frame rendered before any camera is set. */
function defaultCamera(): CameraState {
  const camera = new PerspectiveCamera(50, 1, 0.1, 1000);
  camera.position.set(5, 5, 5);
  camera.lookAt(0, 0, 0);

  return serializeCamera(camera);
}
