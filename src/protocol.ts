import type { SerializedScene } from "./sceneSerialization.js";
import type { Bucket } from "./rasterizer.js";
import type { CpuEnvironmentSnapshot } from "./environment.js";
import type { PreparedGeometry } from "./preparedGeometry.js";
import type { FrameAov } from "./denoise.js";
import type { IrradianceGridBuffers } from "./irradianceCache.js";
import type { FrameSettings } from "./frameSettings.js";
import type { CameraState, RenderEnvironment } from "./settings.js";

// Messages between `CpuRenderer` and its workers. Internal: they change without notice between versions.

/** Worker frame phases: the whole-image irradiance cache grid first, then bucket shading. */
export type FramePhase = "cache" | "shade";

/** Everything the preparing worker publishes so the others can render the same revision. */
export interface PreparedFrame {
  scene: SerializedScene;
  camera: CameraState;
  settings: FrameSettings;
  width: number;
  height: number;
  environment: CpuEnvironmentSnapshot;
  geometry: PreparedGeometry;
  /** Denoiser buffers; absent when the frame did not request it. */
  aov?: FrameAov;
  /** Whole-image irradiance cache record grid; absent without a cache. */
  grid?: IrradianceGridBuffers;
}

/** Packed accumulations of one bucket, when the page does not share memory between workers. */
export interface BucketAovSlice {
  bucket: Bucket;
  data: Float32Array<ArrayBuffer>;
}

/**
 * Protocol between the client and each pool worker. Every `frame` advances the revision and goes only to the first
 * worker, which prepares; the others receive `reset` and then `adopt` once it reports `prepared`. `render` hands a
 * bucket to a specific worker after that worker has reported `start`.
 */
export type WorkerRequest = {
  revision: number;
} & ({
  /** The inputs of a new frame. Scene and environment travel only when they changed since the previous frame. */
  type: "frame";
  /** Readers exist: post the prepared frame back so they can adopt it. A single worker keeps it locally. */
  publishPrepared: boolean;
  /** Image size in CSS pixels; the render size also applies `pixelRatio` and `settings.renderScale`. */
  width: number;
  height: number;
  pixelRatio: number;
  settings: FrameSettings;
  camera: CameraState;
  scene?: SerializedScene;
  environment?: RenderEnvironment;
} | {
  type: "adopt";
  frame: PreparedFrame;
} | {
  type: "reset";
} | {
  /** Work for this worker: a band of the cache grid (`phase: "cache"`) or a bucket to shade. */
  type: "render";
  bucket: number;
  phase: FramePhase;
} | {
  /** Without shared memory: records computed by other workers, for the local grid. */
  type: "records";
  cells: Float32Array<ArrayBuffer>;
} | {
  type: "pause";
  paused: boolean;
} | {
  /** All buckets done: the first worker denoises the indirect light and resends the image in bands. */
  type: "denoise";
  slices: BucketAovSlice[];
} | {
  type: "dispose";
});

export type WorkerResponse = {
  revision: number;
} & ({
  type: "prepared";
  frame: PreparedFrame;
} | {
  type: "start";
  width: number;
  height: number;
  /** Grid indices of the buckets, already ordered center-out; `render` refers to these indices. */
  order: number[];
  /** The frame ends with a denoising pass after the last bucket. */
  denoise: boolean;
  /** Cache-phase jobs per level, coarse to fine; empty when the frame has no cache. */
  prepass: number[];
} | {
  /** A finished band of the cache grid; `cells` is only sent without shared memory. */
  type: "cached";
  job: number;
  cells?: Float32Array<ArrayBuffer>;
} | {
  type: "bucket";
  bucket: Bucket;
  pixels: Uint8ClampedArray<ArrayBuffer>;
  aov?: Float32Array<ArrayBuffer>;
} | {
  type: "denoised";
} | {
  /** Rows of the in-progress bucket starting at `row`, with the pixels resolved so far; missing ones are empty. */
  type: "pixels";
  bucket: Bucket;
  row: number;
  pixels: Uint8ClampedArray<ArrayBuffer>;
} | {
  type: "active";
  bucket: Bucket;
} | {
  type: "error";
  message: string;
});
