import { cameraProjection } from "./camera.js";
import { Color, Matrix3, Matrix4 } from "three";
import type { SerializedFog, SerializedMaterial, SerializedScene } from "./sceneSerialization.js";
import type { CameraState } from "./settings.js";
import type { FrameSettings } from "./frameSettings.js";
import { ShadowBvh, type RayHit } from "./bvh.js";
import { DiffuseGi } from "./globalIllumination.js";
import {
  GRID_REACH, IrradianceCache, IrradianceGrid, type IrradianceGridBuffers, LOOKUP_SMOOTHING, PLACEMENT_COVERAGE,
} from "./irradianceCache.js";
import type { FrameAov } from "./denoise.js";
import { CpuVolumetricLight } from "./volumetricLight.js";
import { type CpuEnvironment } from "./environment.js";
import { matrixValue, type Value } from "./glsl.js";
import { CpuMaterial, type LightingContext, type ShadedFragment, type Surface } from "./material.js";
import { add, clamp, cross, dot, halton, mix, mul, normalize, scale, sub, transform3, transform4 } from "./math.js";
import { encodeColor, toneMap3 } from "./color.js";
import { linearToSrgb } from "./texture.js";
import {
  CAST_SHADOW,
  FLAT_NORMAL,
  GeometryWriter,
  interpolateStored,
  readValue,
  RECEIVE_SHADOW,
  SHADOW_OPAQUE,
  VERTEX_HEADER,
  vertexLayout,
  writeValue,
  type PreparedGeometry,
  type VertexLayout,
} from "./preparedGeometry.js";

export interface Bucket {
  x: number;
  y: number;
  width: number;
  height: number;
  index: number;
}

export interface RasterVertex {
  pointSize?: number;
  clip: number[];
  world: number[];
  normal: number[];
  attributes: Record<string, number[]>;
  varyings: Record<string, Value>;
}

/** What each worker builds per draw group: compiled material and mesh data for this camera. */
interface RuntimeGroup {
  material: CpuMaterial;
  uniforms: Record<string, Value>;
  renderOrder: number;
  winding: number;
  layout: VertexLayout;
}

interface TransparentEntry {
  fragment: ShadedFragment;
  material: CpuMaterial;
  order: number;
  mask: number;
  /** Per-sample depth, or null when the shader wrote gl_FragDepth (then `depth` applies). */
  depths: Float64Array | null;
  depth: number;
}

/** Per-pixel accumulators of a bucket for the denoiser; `indirect` is already weighted by `weight`. */
interface BucketAov {
  modulation: Float32Array;
  indirect: Float32Array;
  weight: Float32Array;
  normal: Float32Array;
  depth: Float32Array;
  variance: Float32Array;
}

const ZERO3 = [0, 0, 0];

/** Fraction of the mean distance to nearby geometry within which an irradiance cache record stays valid. */
const CACHE_ACCURACY = .35;

/** Record validity radius in grid steps: never below one step, nor beyond the lookup window. */
const CACHE_MIN_SPACING = 1.5, CACHE_MAX_SPACING = GRID_REACH;

/** Cache prepass levels, in grid steps: coarse to fine, each one only where the previous level does not reach. */
const PREPASS_LEVELS = [4, 2, 1];

/** Approximate height, in pixels, of the image band covered by one cache prepass job. */
const PREPASS_BAND = 32;

/** One cache prepass job: rows of one grid level, spanning the full image width. */
export interface PrepassJob {
  level: number;
  row: number;
  rows: number;
  /** Covered pixel band, for the preview and the activity indicator. */
  y: number;
  height: number;
}

/** Rays per fragment for ambient occlusion and soft shadows; a per-pixel rotation scatters the pattern. */
const AO_SAMPLES = 12, SHADOW_SAMPLES = 8, TRANSMISSION_RAYS = 64;

const smoothstep = (a: number, b: number, x: number) => {
  if (b <= a)
    return x >= b ? 1 : 0;
  const t = clamp((x - a) / (b - a));

  return t * t * (3 - 2 * t);
};

/** Deterministic noise per pixel position, in [0, 1). */
const hash = (x: number, y: number) => {
  const v = Math.sin(x * 12.9898 + y * 78.233) * 43758.5453;

  return v - Math.floor(v);
};

/** Orthonormal basis around a unit direction. */
function frame(n: number[]) {
  const t = normalize(cross(Math.abs(n[1]) > .99 ? [1, 0, 0] : [0, 1, 0], n));

  return { t, b: cross(n, t) };
}

const shadowThreshold = (material: CpuMaterial) => Math.max(material.number("alphaTest", 0), .5);

function interpolate3(a: Value, b: Value, c: Value, w0: number, w1: number, w2: number): Value {
  if (typeof a === "number")
    return a * w0 + Number(b) * w1 + Number(c) * w2;
  if (Array.isArray(a))
    return a.map((v, i) => interpolate3(v, (b as Value[])[i], (c as Value[])[i], w0, w1, w2));

  return a;
}

interface ScreenTriangle {
  vertices: RasterVertex[];
  xy: number[][];
  area: number;
  front: boolean;
  source: number;
}

export interface BucketResult {
  bucket: Bucket;
  pixels: Uint8ClampedArray;
  linear?: Float32Array;
  depth?: Float32Array;
}

/** Constants of a projected triangle, used at every pixel it covers. */
function passTriangle(triangle: ScreenTriangle, material: CpuMaterial) {
  const [p0, p1, p2] = triangle.xy, [v0, v1, v2] = triangle.vertices;
  const invW0 = 1 / v0.clip[3], invW1 = 1 / v1.clip[3], invW2 = 1 / v2.clip[3];
  const depthTest = material.flag("depthTest", true);
  const z0 = v0.clip[2] * invW0 * .5 + .5, z1 = v1.clip[2] * invW1 * .5 + .5, z2 = v2.clip[2] * invW2 * .5 + .5;
  let polygonOffset = 0;

  if (material.flag("polygonOffset")) {
    // glPolygonOffset: factor scales the triangle's steepest depth slope per pixel, units the depth resolution.
    const ax = p1[0] - p0[0], ay = p1[1] - p0[1], bx = p2[0] - p0[0], by = p2[1] - p0[1], det = ax * by - bx * ay;

    const slope = det ? Math.max(Math.abs(((z1 - z0) * by - (z2 - z0) * ay) / det),
      Math.abs(((z2 - z0) * ax - (z1 - z0) * bx) / det)) : 0;

    polygonOffset = material.number("polygonOffsetFactor", 0) * slope
      + material.number("polygonOffsetUnits", 0) / 16777216;
  }

  return {
    triangle,
    tl0: topLeft(p1, p2), tl1: topLeft(p2, p0), tl2: topLeft(p0, p1),
    translucent: material.flag("transparent") || material.number("transmission", 0) > 0,
    wireframe: material.flag("wireframe"), lineWidth: material.number("wireframeLinewidth", 1),
    depthTest, depthFunc: material.number("depthFunc", 3),
    depthWrite: material.flag("depthWrite", true), colorWrite: material.flag("colorWrite", true),
    polygonOffset,
    earlyZ: !material.fragment && depthTest,
    invW0, invW1, invW2,
    z0, z1, z2,
    invArea: 1 / triangle.area,
    len0: Math.hypot(p2[0] - p1[0], p2[1] - p1[1]), len1: Math.hypot(p0[0] - p2[0], p0[1] - p2[1]),
    len2: Math.hypot(p1[0] - p0[0], p1[1] - p0[1]),
  };
}

type PassTriangle = ReturnType<typeof passTriangle>;

/** Whether any sample of the pixel falls inside the triangle. No depth test, since that depends on draw order. */
function coversPixel(t: PassTriangle, x: number, y: number, jitters: number[][]) {
  const [p0, p1, p2] = t.triangle.xy;

  for (const [jx, jy] of jitters) {
    const px = x + jx, py = y + jy;
    const e0 = edge(p1, p2, px, py), e1 = edge(p2, p0, px, py), e2 = edge(p0, p1, px, py);
    if (e0 < 0 || e1 < 0 || e2 < 0 || (e0 === 0 && !t.tl0) || (e1 === 0 && !t.tl1) || (e2 === 0 && !t.tl2))
      continue;
    if (t.wireframe && Math.min(e0 / t.len0, e1 / t.len1, e2 / t.len2) > t.lineWidth)
      continue;

    return true;
  }

  return false;
}

export function createBuckets(width: number, height: number, size: number): Bucket[] {
  const buckets: Bucket[] = [];
  for (let y = 0; y < height; y += size)
    for (let x = 0; x < width; x += size)
      buckets.push({

        x,
        y,
        width: Math.min(size, width - x),
        height: Math.min(size, height - y),
        index: buckets.length,

      });

  // Center-out ordering, with stable row order for equal distances.
  return buckets.sort((a, b) =>
    (a.x + a.width / 2 - width / 2) ** 2 + (a.y + a.height / 2 - height / 2) ** 2
    - ((b.x + b.width / 2 - width / 2) ** 2 + (b.y + b.height / 2 - height / 2) ** 2));
}

function interpolateValue(a: Value, b: Value, t: number): Value {
  if (typeof a === "number" && typeof b === "number")
    return a + (b - a) * t;
  if (Array.isArray(a) && Array.isArray(b))
    return a.map((v, i) => interpolateValue(v, b[i], t));

  return a;
}

function interpolateVertex(a: RasterVertex, b: RasterVertex, t: number): RasterVertex {
  const lerp = (x: number[], y: number[]) => x.map((v, i) => v + (y[i] - v) * t);

  return {
    clip: lerp(
      a.clip,
      b.clip,
    ),
    world: lerp(
      a.world,
      b.world,
    ),
    normal: lerp(
      a.normal,
      b.normal,
    ),
    attributes: Object.fromEntries(Object.entries(a.attributes).map(([k, v]) => [k, lerp(
      v,
      b.attributes[k],
    )])),
    varyings: Object.fromEntries(Object.entries(a.varyings).map(([k, v]) => [k, interpolateValue(
      v,
      b.varyings[k],
      t,
    )])),
  };
}

/** Sutherland-Hodgman in homogeneous coordinates; attributes are clipped before perspective division. */
export function clipPolygon(vertices: RasterVertex[], planes = [0, 1, 2, 3, 4, 5]): RasterVertex[] {
  let polygon = vertices;
  const distance = (v: RasterVertex, p: number) => v.clip[3] + v.clip[Math.floor(p / 2)] * (p % 2 === 0 ? 1 : -1);

  for (const plane of planes) {
    const output: RasterVertex[] = [];

    for (let i = 0; i < polygon.length; i++) {
      const a = polygon[i], b = polygon[(i + 1) % polygon.length], da = distance(
          a,
          plane,
        ), db = distance(
          b,
          plane,
        );

      if (da >= 0)
        output.push(a);
      if ((da >= 0) !== (db >= 0))
        output.push(interpolateVertex(a, b, da / (da - db)));
    }

    polygon = output;
    if (!polygon.length)
      break;
  }

  return polygon;
}

const edge = (a: number[], b: number[], x: number, y: number) =>
  (b[0] - a[0]) * (y - a[1]) - (b[1] - a[1]) * (x - a[0]);

const topLeft = (a: number[], b: number[]) => b[1] < a[1] || (b[1] === a[1] && b[0] > a[0]);

export function depthPass(func: number, incoming: number, stored: number) {
  switch (func) {
    case 0: return false;
    case 1: return true;
    case 2: return incoming < stored;
    case 3: return incoming <= stored;
    case 4: return incoming === stored;
    case 5: return incoming >= stored;
    case 6: return incoming > stored;
    case 7: return incoming !== stored;
    default: return incoming <= stored;
  }
}

export interface RasterTargetOptions {
  aspect?: number;
  linear?: boolean;
  clipPlane?: number[];
}

export class CpuRasterizer {
  readonly buckets: Bucket[];
  private groups: RuntimeGroup[] = [];
  private geometry: PreparedGeometry | null = null;
  private bvh: ShadowBvh | null = null;
  private gi: DiffuseGi | null = null;
  // Indirect light records of the bucket in progress; null when the cache is disabled.
  private giCache: IrradianceCache | null = null;
  // The bucket's grid is already computed (in this call or in an earlier prepass): do not place it again.
  private cachePlaced = false;
  // Full-image record grid, built by the cache prepass; null when there is no separate prepass.
  private giGrid: IrradianceGrid | null = null;
  // Pixel of the fragment being shaded, to look up the right grid window.
  private fragmentX = 0;
  private fragmentY = 0;
  // Variance of the last indirect light lookup, read right after shading the fragment.
  private lastVariance = 0;
  // Full-image buffers for the denoiser, when this frame requested it.
  private aov: FrameAov | null = null;
  // Direction of the ray in flight: the soft shadow and occlusion loops overwrite the three values on
  // each sample instead of allocating a vector per ray. `trace` reads the components on entry and does
  // not keep the reference, so reusing the same buffer is safe.
  private readonly rayDirection: number[] = [0, 0, 0];
  // Tangent frame of the last requested triangle: neighboring fragments usually belong to the same one.
  private tangentSource = -1;
  private tangent = ZERO3;
  private bitangentSign = 1;
  // Vertices read from the geometry during a bucket, keyed by offset: reused by neighboring triangles and lens passes.
  private vertexCache: Map<number, RasterVertex> | null = null;
  // Flat-shaded triangles replace the normal of all three vertices, so they are cached per triangle.
  private flatCache: Map<number, RasterVertex[]> | null = null;
  private projection: number[];
  private inverseProjection: number[];
  private volumetric: CpuVolumetricLight;
  private view: number[];
  private cameraPosition: number[];
  private cameraWorld: number[];
  private cameraHandedness: number;
  private lensRadius: number;
  private backgroundColor: number[];
  /** The render settings' fog covers everything; otherwise `scene.fog` applies to the materials that accept it. */
  private fog: (SerializedFog & { everything: boolean }) | null;
  private globals: Record<string, Value>;

  /**
   * `adopted` is the geometry another worker already prepared for the same scene, camera, settings and resolution:
   * with it, `prepare` only compiles the materials and reads the shared arrays.
   */
  constructor(
    readonly scene: SerializedScene,
    readonly camera: CameraState,
    readonly settings: FrameSettings,
    readonly width: number,
    readonly height: number,
    readonly environment: CpuEnvironment,
    readonly target: RasterTargetOptions = {},
    private readonly adopted?: PreparedGeometry) {
    if (!Number.isInteger(width) || !Number.isInteger(height) || width < 1 || height < 1 || width * height > 32000000)
      throw new Error("CPU renderer: invalid resolution (maximum 32 megapixels).");
    if (!Number.isInteger(settings.tileSize) || settings.tileSize < 8 || settings.tileSize > 256
      || !Number.isInteger(settings.maxSamples)
      || settings.maxSamples < 1
      || settings.maxSamples > 256)
      throw new Error("CPU renderer: invalid sampling or bucket settings.");
    if (settings.globalIllumination && (
      !Number.isInteger(settings.giSamples ?? 64)
      || (settings.giSamples ?? 64) < 1 || (settings.giSamples ?? 64) > 1024
      || !Number.isInteger(settings.giBounces ?? 3) || (settings.giBounces ?? 3) < 1 || (settings.giBounces ?? 3) > 8
      || !Number.isFinite(settings.giIntensity ?? 1)
      || (settings.giIntensity ?? 1) < 0 || (settings.giIntensity ?? 1) > 4
      || !Number.isFinite(settings.giClamp ?? 0) || (settings.giClamp ?? 0) < 0 || (settings.giClamp ?? 0) > 100
      || !Number.isInteger(settings.giCacheSpacing ?? 4) || (settings.giCacheSpacing ?? 4) < 1
      || (settings.giCacheSpacing ?? 4) > 16))
      throw new Error("CPU renderer: invalid global illumination settings.");
    const projection = cameraProjection(camera, target.aspect ?? width / height);
    this.projection = projection.toArray();
    this.inverseProjection = projection.clone().invert().toArray();
    this.volumetric = new CpuVolumetricLight(scene.lights);
    this.cameraWorld = camera.matrixWorld;
    this.cameraHandedness = Math.sign(new Matrix4().fromArray(this.cameraWorld).determinant());
    this.view = new Matrix4().fromArray(camera.matrixWorld).invert().toArray();
    this.cameraPosition = camera.matrixWorld.slice(12, 15);
    this.buckets = createBuckets(width, height, settings.tileSize);
    const focalLength = .035 / (2 * Math.tan((camera.fov ?? 50) * Math.PI / 360));
    this.lensRadius = settings.dofEnabled && !this.orthographic ? focalLength / (2 * settings.dofAperture) : 0;
    this.backgroundColor = new Color(settings.backgroundColor).toArray();
    this.fog = settings.fogEnabled ? { color: new Color(settings.fogColor ?? "#ffffff").toArray(),
      near: settings.fogNear ?? 0, far: settings.fogFar ?? 1, everything: true }
      : scene.fog ? { ...scene.fog, everything: false } : null;
    this.globals = {

      projectionMatrix: matrixValue(this.projection),
      viewMatrix: matrixValue(this.view),
      cameraPosition: this.cameraPosition,
      isOrthographic: this.orthographic,
      cameraNear: camera.near,
      cameraFar: camera.far,
      directionalLights: scene.lights.filter(l => l.type === "DirectionalLight").map(l => ({
        color: scale(l.color, l.intensity),
        direction: normalize(transform4(this.view, [...normalize(sub(l.position, l.target)), 0]).slice(0, 3)),
      })),
      pointLights: scene.lights.filter(l => l.type === "PointLight").map(l => ({
        color: scale(l.color, l.intensity), position: transform4(this.view, l.position).slice(0, 3),
        distance: l.distance, decay: l.decay,
      })),
      spotLights: scene.lights.filter(l => l.type === "SpotLight").map(l => ({
        color: scale(l.color, l.intensity), position: transform4(this.view, l.position).slice(0, 3),
        distance: l.distance, decay: l.decay,
        direction: normalize(transform4(this.view, [...normalize(sub(l.position, l.target)), 0]).slice(0, 3)),
        coneCos: Math.cos(l.angle), penumbraCos: Math.cos(l.angle * (1 - l.penumbra)),
      })),
    };
  }

  /** Attaches the denoiser buffers; each finished bucket writes its own region into them. */
  attachAov(aov: FrameAov) {
    if (aov.width !== this.width || aov.height !== this.height)
      throw new Error("CPU renderer: denoise buffers do not match the frame.");
    this.aov = aov;
  }

  get frameAov(): FrameAov | null {
    return this.aov;
  }

  /** Whether this frame runs an irradiance cache prepass before shading. */
  get usesCache(): boolean {
    return Boolean(this.gi && this.settings.giCache);
  }

  /**
   * Distance, in pixels, over which neighboring pixels share the noise of their indirect light: the reach of a
   * cache record. Zero without the cache, where every pixel has its own estimate. The denoiser needs it.
   */
  get noiseCorrelation(): number {
    return this.usesCache ? (this.settings.giCacheSpacing ?? 4) * CACHE_MAX_SPACING * LOOKUP_SMOOTHING : 0;
  }

  /** Attaches the full-image record grid that the cache prepass fills and shading looks up. */
  attachIrradianceGrid(buffers: IrradianceGridBuffers) {
    if (buffers.width !== this.width || buffers.height !== this.height)
      throw new Error("CPU renderer: irradiance grid does not match the frame.");
    this.giGrid = new IrradianceGrid(buffers);
  }

  get irradianceGrid(): IrradianceGrid | null {
    return this.giGrid;
  }

  /** Cache prepass jobs, per level: the client and the worker compute the same list. */
  static prepassJobs(_width: number, height: number, spacing: number): PrepassJob[][] {
    return PREPASS_LEVELS.map(level => {
      const step = level * spacing, gridRows = Math.ceil(height / step),
        perJob = Math.max(1, Math.floor(PREPASS_BAND / step));

      const jobs: PrepassJob[] = [];

      for (let row = 0; row < gridRows; row += perJob) {
        const rows = Math.min(perJob, gridRows - row), y = row * step;
        jobs.push({ level, row, rows, y, height: Math.min(height - y, rows * step) });
      }

      return jobs;
    });
  }

  prepassJob(index: number): PrepassJob {
    const job = CpuRasterizer.prepassJobs(this.width, this.height, this.settings.giCacheSpacing ?? 4).flat()[index];
    if (!job)
      throw new Error("CPU renderer: unknown irradiance cache job.");

    return job;
  }

  /**
   * Opaque surface visible at a pixel center, via a camera ray: the first one that passes the same rules as
   * record placement (opaque, depth tested, writes color, queries indirect light, correct side, alpha).
   */
  private primarySurface(px: number, py: number): { source: number; surface: Surface } | null {
    if (!this.bvh)
      return null;

    const { direction, origin: rayOrigin } = this.viewRay(px, py);

    let origin = rayOrigin, ignore = -1;

    for (let layer = 0; layer < 64; layer++) {
      const hit = this.bvh.intersect(origin, direction, Infinity, ignore);
      if (!hit)
        return null;
      const material = this.group(hit.index).material, surface = this.raySurface(hit, direction);
      const side = material.number("side", 0);

      const eligible = !material.flag("transparent") && material.number("transmission", 0) <= 0
        && material.flag("depthTest", true) && material.flag("colorWrite", true) && material.diffuseGi
        && (side === 2 || (side === 1 ? !surface.frontFacing : surface.frontFacing));

      if (eligible) {
        const alphaTest = material.number("alphaTest", 0);
        let alpha = alphaTest > 0 ? material.alpha(surface.attributes, material.uvGradients(surface)) : 1;
        if (alphaTest > 0 && material.flag("vertexColors") && surface.attributes.color)
          alpha *= surface.attributes.color[3] ?? 1;
        if (alpha >= alphaTest)
          return { source: hit.index, surface };
      }

      origin = this.rayOrigin(surface.position, direction);
      ignore = hit.index;
    }

    return null;
  }

  /**
   * Opaque scene the camera sees through a world-space point: what Three reads from its transmission framebuffer
   * at the refracted exit. Translucent surfaces are skipped, as they are absent from that pass; `blur` spreads the
   * ray over the footprint of the mip level Three would sample for that roughness.
   */
  private transmittedLight(
    exit: number[],
    blur: number,
    seed: number,
    shade: (source: number, surface: Surface, px: number, py: number) => ShadedFragment | null): number[] | undefined {
    const clip = transform4(this.projection, transform4(this.view, exit));
    if (!this.bvh || clip[3] <= 1e-10)
      return undefined;
    // Three reads mip level log2(width) * blur bicubically: a spread of about 0.65 of that level's texel, in pixels.
    const spread = blur > 0 ? .65 * this.width ** blur : 0;
    const cx = (clip[0] / clip[3] * .5 + .5) * this.width, cy = (.5 - clip[1] / clip[3] * .5) * this.height;
    // A wide spread averages several Gaussian-distributed rays; each pixel rotates the pattern.
    const rays = spread > 1 ? Math.min(TRANSMISSION_RAYS, Math.ceil(spread * spread)) : 1, sum = [0, 0, 0];

    for (let i = 0; i < rays; i++) {
      const u = (halton(i + 1, 2) + seed) % 1, v = (halton(i + 1, 3) + seed * .618034) % 1;
      const radius = spread * Math.sqrt(-2 * Math.log(1 - u * .999)), angle = 2 * Math.PI * v;
      const light = this.cameraRay(cx + radius * Math.cos(angle), cy + radius * Math.sin(angle), shade);
      sum[0] += light[0]; sum[1] += light[1]; sum[2] += light[2];
    }

    return scale(sum, 1 / rays);
  }

  /** Opaque scene along the camera ray through image position `px`, `py`, skipping translucent surfaces. */
  private cameraRay(px: number, py: number,
    shade: (source: number, surface: Surface, px: number, py: number) => ShadedFragment | null): number[] {
    const { direction, origin: rayOrigin } = this.viewRay(px, py);

    const clipPlane = this.target.clipPlane;
    let origin = rayOrigin, ignore = -1;

    for (let layer = 0; layer < 64; layer++) {
      const hit = this.bvh!.intersect(origin, direction, Infinity, ignore);
      if (!hit)
        break;
      const material = this.group(hit.index).material, surface = this.raySurface(hit, direction);
      const side = material.number("side", 0);

      const eligible = !material.flag("transparent") && material.number("transmission", 0) <= 0
        && material.flag("colorWrite", true) && (side === 2 || (side === 1 ? !surface.frontFacing : surface.frontFacing))
        && !(clipPlane && dot(clipPlane, surface.position) + clipPlane[3] < 0);

      const fragment = eligible ? shade(hit.index, surface, px, py) : null;

      if (fragment) {
        this.applyFog(fragment, surface.position, material);

        return fragment.color;
      }

      origin = this.rayOrigin(surface.position, direction);
      ignore = hit.index;
    }

    return this.background(px, py).slice(0, 3);
  }

  /** How much of the fog color replaces what lies at `position`: Three's Fog or FogExp2 over view depth. */
  private fogAmount(position: number[]) {
    const fog = this.fog!, depth = this.viewDepth(position);

    return fog.density !== undefined ? 1 - Math.exp(-((fog.density * depth) ** 2))
      : smoothstep(fog.near, fog.far, depth);
  }

  /** Like Three's Fog and FogExp2: over view depth, in linear light, leaving the background untouched. */
  private applyFog(fragment: ShadedFragment, position: number[], material: CpuMaterial) {
    const fog = this.fog;
    if (!fog || !(fog.everything || material.flag("fog", true)))
      return;
    const amount = this.fogAmount(position);
    if (amount <= 0)
      return;
    fragment.color = mix(fragment.color, fog.color, amount);
    fragment.diffuse = mix(fragment.diffuse, fog.color, amount);
    if (fragment.modulation)
      fragment.modulation = scale(fragment.modulation, 1 - amount);
  }

  /**
   * One cache prepass job: walks the level's grid points in the band, finds the surface by ray tracing,
   * skips points the grid already covers and computes the rest, writing into the shared grid. Each record becomes
   * a gray dot in the preview (`progress` receives the band buffer and the pixel index within it).
   * Returns the written cells.
   */
  *cachePrepass(
    index: number,
    progress?: (pixels: Uint8ClampedArray, pixel: number) => void): Generator<void, number[]> {
    const grid = this.giGrid, written: number[] = [];
    if (!grid || !this.gi)
      return written;
    const job = this.prepassJob(index), spacing = grid.buffers.spacing, step = job.level * spacing;
    const columns = Math.ceil(this.width / step), coarsest = job.level === PREPASS_LEVELS[0];
    const pixels = progress ? new Uint8ClampedArray(this.width * job.height * 4) : null;
    const dot = Math.max(1, Math.min(2, spacing)), preview = new Float32Array(4);
    const vignette = clamp(this.settings.vignette ?? 0), grain = clamp(this.settings.grain ?? 0);
    let lastYield = performance.now();

    for (let gy = job.row; gy < job.row + job.rows; gy++)
      for (let gx = 0; gx < columns; gx++) {
        // Points already computed at the coarser level (multiples of its step) are not repeated.
        if (!coarsest && gx % 2 === 0 && gy % 2 === 0)
          continue;
        const x = gx * step, y = gy * step;
        if (x >= this.width || y >= this.height)
          continue;
        const cx = gx * job.level, cy = gy * job.level;
        if (grid.has(cx, cy))
          continue;
        const px = x + .5, py = y + .5, primary = this.primarySurface(px, py);
        if (!primary)
          continue;
        const normal = normalize(primary.surface.normal);
        // Adaptive: where existing records already cover the point well, the finer level costs nothing.
        if (grid.lookupAt(px, py, primary.surface.position, normal, PLACEMENT_COVERAGE))
          continue;
        const estimate = this.gi.estimate(primary.source, primary.surface.position, normal, hash(px, py));
        const footprint = this.pixelFootprint(this.viewDepth(primary.surface.position)) * spacing;

        const radius = clamp(
          estimate.radius * CACHE_ACCURACY, footprint * CACHE_MIN_SPACING, footprint * CACHE_MAX_SPACING);

        written.push(grid.write(cx, cy, { position: primary.surface.position, normal, color: estimate.color,
          variance: estimate.variance, radius }));

        if (pixels && progress) {
          // Grayscale preview: the indirect light luminance, with a floor so every dot stays visible.
          const l = .25 + .75 * Math.min(1,
            estimate.color[0] * .2126 + estimate.color[1] * .7152 + estimate.color[2] * .0722);

          preview[0] = l; preview[1] = l; preview[2] = l; preview[3] = 1;

          for (let dy = 0; dy < dot; dy++)
            for (let dx = 0; dx < dot; dx++) {
              const sx = x + dx, sy = y - job.y + dy;
              if (sx >= this.width || sy >= job.height)
                continue;
              const i = sy * this.width + sx;
              this.resolveSample(sx + .5, y + dy + .5, preview, 0, 1, pixels, i * 4, vignette, grain);
              progress(pixels, i);
            }
        }

        if (performance.now() - lastYield > 8) {
          yield;
          lastYield = performance.now();
        }
      }

    return written;
  }

  /** Prepared geometry, for the other workers to adopt. */
  get preparedGeometry(): PreparedGeometry {
    if (!this.geometry)
      throw new Error("CPU renderer: geometry is not prepared yet.");

    return this.geometry;
  }

  /** Incremental preparation lets the worker process cancellation during large scene exports. */
  *prepare(): Generator<void> {
    const textures = new Map(this.scene.textures.map(t => [t.id, t]));
    const materials = new Map<object, CpuMaterial>();
    const meshes = new Map<number, { uniforms: Record<string, Value>; winding: number; worldNormal: number[] }>();

    const meshContext = (index: number) => {
      let context = meshes.get(index);

      if (!context) {
        const mesh = this.scene.meshes[index];
        const model = new Matrix4().fromArray(mesh.matrixWorld);
        const modelView = new Matrix4().fromArray(this.view).multiply(model);
        const viewNormal = new Matrix3().getNormalMatrix(modelView).toArray();
        context = {
          winding: Math.sign(model.determinant() * new Matrix4().fromArray(this.cameraWorld).determinant()),
          worldNormal: new Matrix3().getNormalMatrix(model).toArray(),
          uniforms: {

            ...this.globals,
            modelMatrix: matrixValue(mesh.matrixWorld),
            modelViewMatrix: matrixValue(modelView.toArray()),
            normalMatrix: matrixValue(viewNormal),

          },
        };
        meshes.set(index, context);
      }

      return context;
    };

    const materialFor = (data: SerializedMaterial) => {
      let material = materials.get(data);

      if (!material) {
        material = new CpuMaterial(data.shader ? { ...data, shader: { ...data.shader, defines: {
          ...data.shader.defines,
          NUM_DIR_LIGHTS: (this.globals.directionalLights as Value[]).length,
          NUM_POINT_LIGHTS: (this.globals.pointLights as Value[]).length,
          NUM_SPOT_LIGHTS: (this.globals.spotLights as Value[]).length,
        } } } : data, textures);
        materials.set(data, material);
      }

      return material;
    };

    if (this.adopted) {
      for (const prepared of this.adopted.groups) {
        const mesh = this.scene.meshes[prepared.mesh];
        const data = mesh?.materials[mesh.groups[prepared.group]?.materialIndex];
        if (!data)
          throw new Error("CPU renderer: prepared geometry does not match the scene.");
        const context = meshContext(prepared.mesh);
        this.groups.push({
          material: materialFor(data),
          uniforms: context.uniforms,
          renderOrder: mesh.renderOrder,
          winding: mesh.primitive ? 1 : context.winding,
          layout: prepared.layout,
        });
        yield;
      }

      this.useGeometry(this.adopted);

      return;
    }

    const writer = new GeometryWriter();
    const bins: number[][] = Array.from({ length: this.buckets.length }, () => []);

    for (let meshIndex = 0; meshIndex < this.scene.meshes.length; meshIndex++) {
      const mesh = this.scene.meshes[meshIndex], context = meshContext(meshIndex), uniforms = context.uniforms;
      const points = mesh.primitive === "points";
      const lines = mesh.primitive === "line" || mesh.primitive === "lineSegments" || mesh.primitive === "lineLoop";

      for (let groupIndex = 0; groupIndex < mesh.groups.length; groupIndex++) {
        const group = mesh.groups[groupIndex];
        const data = mesh.materials[group.materialIndex];
        if (!data)
          throw new Error("CPU renderer: invalid geometry material group.");
        const material = materialFor(data);
        if (!material.flag("visible", true))
          continue;
        let layout: VertexLayout | null = null, runtimeIndex = -1;
        const vertexCache = new Map<number, number>();

        const vertex = (index: number): number => {
          const cached = vertexCache.get(index);
          if (cached !== undefined)
            return cached;
          const attributes: Record<string, number[]> = {};
          for (const [name, attribute] of Object.entries(mesh.attributes))
            attributes[name]
              = Array.from(attribute.data.subarray(
                index * attribute.itemSize,
                (index + 1) * attribute.itemSize,
              ));
          let position = attributes.position, normal = attributes.normal ?? [0, 0, 1];
          if (position.length !== 3)
            throw new Error("CPU renderer: invalid position attribute.");
          if (material.data.maps.displacementMap)
            position
              = add(
                position,
                scale(
                  normal,
                  material.map(
                    "displacementMap",
                    attributes,
                  )[0] * material.number(
                    "displacementScale",
                    1,
                  )
                  + material.number(
                    "displacementBias",
                    0,
                  ),
                ),
              );
          let clip: number[] | undefined;
          let pointSize = 1;
          const varyings: Record<string, Value> = {};

          if (material.vertex) {
            const result = material.vertex.run({

              ...uniforms,
              ...material.uniforms,
              ...Object.fromEntries(Object.entries(attributes).map(([name, value]) =>
                [name, value.length === 1 ? value[0] : value])),
              uPointScale: this.height / (2 * Math.tan((this.camera.fov ?? 50) * Math.PI / 360)),
              gl_PointSize: 1,
              position,
              normal,
              transformed: position,
              objectNormal: normal,
              cpuPosition: position,
              cpuNormal: normal,
              gl_Position: [NaN, NaN, NaN, NaN],
              gl_VertexID: index,

            }, { texture: material.texture });

            if (!result)
              throw new Error(`${data.name}: discard is invalid in a vertex shader.`);
            if (material.data.shader?.patched) {
              position = result.cpuPosition as number[];
              normal = result.cpuNormal as number[];
              if ((result.gl_Position as number[]).every(Number.isFinite))
                clip = result.gl_Position as number[];
            } else
              clip = result.gl_Position as number[];
            pointSize = Number(result.gl_PointSize ?? 1);
            for (const name of material.vertex.varyings)
              if (result[name] !== undefined)
                varyings[name] = result[name];
          }

          let world = transform4(mesh.matrixWorld, position).slice(0, 3);

          if (mesh.primitive === "sprite" && !material.vertex) {
            const center = transform4(this.view, mesh.matrixWorld.slice(12, 15));
            const attenuation = !this.orthographic && !material.flag("sizeAttenuation", true) ? -center[2] : 1;
            const sx = Math.hypot(...mesh.matrixWorld.slice(0, 3)) * attenuation;
            const sy = Math.hypot(...mesh.matrixWorld.slice(4, 7)) * attenuation;
            const x = (position[0] - ((mesh.center?.[0] ?? .5) - .5)) * sx;
            const y = (position[1] - ((mesh.center?.[1] ?? .5) - .5)) * sy;
            const angle = material.number("rotation", 0), c = Math.cos(angle), s = Math.sin(angle);
            center[0] += x * c - y * s; center[1] += x * s + y * c;
            clip = transform4(this.projection, center);
          }

          if (points && !material.vertex) {
            pointSize = material.number("size", 1);
            if (!this.orthographic && material.flag("sizeAttenuation", true))
              pointSize *= this.height / (2 * -transform4(this.view, world)[2]);
          }

          if (clip) {
            const viewPosition = transform4(this.inverseProjection, clip);
            world = transform4(this.cameraWorld, viewPosition.map(v => v / viewPosition[3])).slice(0, 3);
          }

          clip ??= transform4(this.projection, transform4(this.view, world));
          if (!clip.every(Number.isFinite))
            throw new Error(`${data.name}: vertex shader produced an invalid gl_Position.`);
          layout ??= vertexLayout(mesh.attributes, varyings, points);
          const offset = writer.allocateVertex(layout.stride), out = writer.vertices;
          out.set(clip.slice(0, 4), offset);
          out.set(world, offset + 4);
          out.set(normalize(transform3(context.worldNormal, normal)), offset + 7);
          out[offset + 10] = pointSize;
          let cursor = offset + VERTEX_HEADER;

          for (const [name, itemSize] of layout.attributes) {
            out.set(attributes[name], cursor);
            cursor += itemSize;
          }

          // The gl_PointCoord at the end of the layout only exists on the corners, written by the point builder.
          const varyingCount = layout.varyings.length - (points ? 1 : 0);
          for (let i = 0; i < varyingCount; i++)
            cursor = writeValue(varyings[layout.varyings[i][0]], layout.varyings[i][1], out, cursor);
          vertexCache.set(index, offset);

          return offset;
        };

        // The group joins the list only when it gets its first triangle, using the first vertex's layout.
        const groupId = () => {
          if (runtimeIndex < 0) {
            runtimeIndex = this.groups.length;
            writer.groups.push({ mesh: meshIndex, group: groupIndex, layout: layout! });
            this.groups.push({
              material, uniforms, renderOrder: mesh.renderOrder,
              winding: mesh.primitive ? 1 : context.winding, layout: layout!,
            });
          }

          return runtimeIndex;
        };

        const shadowOpaque = !material.data.maps.map && !material.data.maps.alphaMap && !material.flag("vertexColors")
          && material.number("opacity", 1) >= shadowThreshold(material);

        const flags = (!mesh.primitive && mesh.castShadow ? CAST_SHADOW : 0) | (mesh.receiveShadow ? RECEIVE_SHADOW : 0)
          | (shadowOpaque ? SHADOW_OPAQUE : 0)
          | (!mesh.primitive && (!mesh.attributes.normal || material.flag("flatShading")) ? FLAT_NORMAL : 0);

        const end = group.start + group.count;
        const step = points || (lines && mesh.primitive !== "lineSegments") ? 1 : lines ? 2 : 3;

        for (let offset = group.start;
          offset + (points || mesh.primitive === "lineLoop" ? 0 : lines ? 1 : 2) < end; offset += step) {
          if (points && offset % 128 === 0) yield;
          const faces: number[][] = [];
          if (lines) {
            if (group.count < 2) continue;
            const next = offset + 1 < end ? offset + 1 : group.start;
            let a = vertex(mesh.index ? mesh.index[offset] : offset);
            let b = vertex(mesh.index ? mesh.index[next] : next);
            // Clip the center line before dividing by w; retain all interpolated attributes (including dash distance).
            let rejected = false;

            for (const sign of [1, -1]) {
              const stored = writer.vertices;
              const da = stored[a + 3] + sign * stored[a + 2], db = stored[b + 3] + sign * stored[b + 2];

              if (da < 0 && db < 0) { rejected = true; break; }

              if (da < 0 || db < 0) {
                const t = da / (da - db), clipped = writer.allocateVertex(layout!.stride), out = writer.vertices;
                for (let c = 0; c < layout!.stride; c++) out[clipped + c] = out[a + c] * (1 - t) + out[b + c] * t;
                if (da < 0) a = clipped; else b = clipped;
              }
            }

            if (rejected) continue;
            const stored = writer.vertices, aw = stored[a + 3], bw = stored[b + 3];
            if (aw <= 0 || bw <= 0) continue;
            const dx = (stored[b] / bw - stored[a] / aw) * this.width;
            const dy = (stored[b + 1] / bw - stored[a + 1] / aw) * this.height;
            const length = Math.hypot(dx, dy), width = material.number("linewidth", 1);
            if (length === 0 || width <= 0) continue;

            const corners = [[a, -1], [b, -1], [b, 1], [a, 1]].map(([source, sign]) => {
              const corner = writer.allocateVertex(layout!.stride), out = writer.vertices;
              out.copyWithin(corner, source, source + layout!.stride);
              out[corner] += -dy / length * width / this.width * out[source + 3] * sign;
              out[corner + 1] += dx / length * width / this.height * out[source + 3] * sign;
              const vp = transform4(this.inverseProjection, Array.from(out.subarray(corner, corner + 4)));
              out.set(transform4(this.cameraWorld, vp.map(v => v / vp[3])).slice(0, 3), corner + 4);

              return corner;
            });

            faces.push([corners[0], corners[1], corners[2]], [corners[0], corners[2], corners[3]]);
          } else if (points) {
            const center = vertex(mesh.index ? mesh.index[offset] : offset), stored = writer.vertices;
            const size = stored[center + 10], w = stored[center + 3], cx = stored[center], cy = stored[center + 1];
            if (size <= 0 || w <= 0) continue;
            if (!Number.isFinite(size)) throw new Error(`${data.name}: invalid gl_PointSize.`);
            const stride = layout!.stride;

            const corners = [[0, 0], [1, 0], [1, 1], [0, 1]].map(([u, v]) => {
              const corner = writer.allocateVertex(stride), out = writer.vertices;
              out.copyWithin(corner, center, center + stride);
              out[corner] = cx + (u * 2 - 1) * size / this.width * w;
              out[corner + 1] = cy + (v * 2 - 1) * size / this.height * w;
              const vp = transform4(this.inverseProjection, Array.from(out.subarray(corner, corner + 4)));
              out.set(transform4(this.cameraWorld, vp.map(v => v / vp[3])).slice(0, 3), corner + 4);
              out[corner + stride - 2] = u;
              out[corner + stride - 1] = 1 - v;

              return corner;
            });

            faces.push([corners[0], corners[1], corners[2]], [corners[0], corners[2], corners[3]]);
          } else faces.push([0, 1, 2].map(i => vertex(mesh.index ? mesh.index[offset + i] : offset + i)));

          for (const face of faces) {
            const source = writer.addTriangle(groupId(), face[0], face[1], face[2], flags);
            const stored = writer.vertices;

            const clipped = clipPolygon(face.map(o => ({
              clip: [stored[o], stored[o + 1], stored[o + 2], stored[o + 3]],
              world: [], normal: [], attributes: {}, varyings: {},
            })), [4, 5]);

            if (clipped.length) {
              let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;

              for (const v of clipped) {
                const w = v.clip[3];
                if (w <= 1e-10)
                  continue;
                const x = (v.clip[0] / w * .5 + .5) * this.width, y = (.5 - v.clip[1] / w * .5) * this.height;

                const blur = Math.abs(
                  this.lensRadius * this.projection[5] * (1 / this.settings.dofFocusDistance - 1 / w) * this.height
                / 2)
                + 1;

                minX = Math.min(minX, x - blur);
                maxX = Math.max(maxX, x + blur);
                minY = Math.min(minY, y - blur);
                maxY = Math.max(maxY, y + blur);
              }

              const columns = Math.ceil(this.width / this.settings.tileSize), tile = this.settings.tileSize;
              for (let row = Math.max(
                0,
                Math.floor(minY / tile),
              ); row <= Math.min(
                  Math.ceil(this.height / tile) - 1,
                  Math.floor(maxY / tile),
                ); row++)
                for (let col = Math.max(
                  0,
                  Math.floor(minX / tile),
                ); col <= Math.min(
                    columns - 1,
                    Math.floor(maxX / tile),
                  ); col++)
                  bins[row * columns + col].push(source);
            }

            if (source % 128 === 0)
              yield;
          }
        }
      }

      yield;
    }

    // Refraction looks up whatever lies behind the glass, shadow caster or not.
    this.useGeometry(writer.finish(bins, Boolean(this.settings.globalIllumination)
      || this.groups.some(group => group.material.number("transmission", 0) > 0)));
  }

  private useGeometry(geometry: PreparedGeometry) {
    this.geometry = geometry;
    this.bvh = new ShadowBvh({ data: geometry.bvh, positions: geometry.positions, castShadow: geometry.flags });
    if (this.settings.globalIllumination) this.gi = new DiffuseGi({
      triangles: {
        count: geometry.flags.length,
        positions: geometry.positions,
        faceNormals: geometry.faceNormals,
        material: index => this.group(index).material,
      },
      settings: this.settings, lights: this.scene.lights, environment: this.environment,
      intersect: (p, d, distance, ignore) => this.bvh!.intersect(p, d, distance, ignore),
      surface: (hit, d) => this.raySurface(hit, d),
      origin: (p, n) => this.rayOrigin(p, n),
    });
  }

  private group(source: number): RuntimeGroup {
    return this.groups[this.geometry!.triangles[source * 4]];
  }

  private hasFlag(source: number, flag: number) {
    return (this.geometry!.flags[source] & flag) !== 0;
  }

  /** A vertex read from the prepared geometry; `normal` replaces the stored one when the face is flat-shaded. */
  private storedVertex(offset: number, layout: VertexLayout, normal: number[] | null): RasterVertex {
    const data = this.geometry!.vertices, attributes: Record<string, number[]> = {};
    let cursor = offset + VERTEX_HEADER;

    for (const [name, size] of layout.attributes) {
      const values = new Array<number>(size);
      for (let i = 0; i < size; i++)
        values[i] = data[cursor + i];
      attributes[name] = values;
      cursor += size;
    }

    const varyings: Record<string, Value> = {}, position = [cursor - offset];
    for (const [name, shape] of layout.varyings)
      varyings[name] = readValue(data, offset, shape, position);

    return {
      clip: [data[offset], data[offset + 1], data[offset + 2], data[offset + 3]],
      pointSize: data[offset + 10],
      world: [data[offset + 4], data[offset + 5], data[offset + 6]],
      normal: normal ?? [data[offset + 7], data[offset + 8], data[offset + 9]],
      attributes,
      varyings,
    };
  }

  /** The three original vertices of a triangle, read from the prepared geometry. */
  private sourceVertices(source: number): RasterVertex[] {
    const { triangles, faceNormals } = this.geometry!, layout = this.group(source).layout;

    if (this.hasFlag(source, FLAT_NORMAL)) {
      const cached = this.flatCache?.get(source);
      if (cached)
        return cached;
      const normal = [faceNormals[source * 3], faceNormals[source * 3 + 1], faceNormals[source * 3 + 2]];
      const vertices = [1, 2, 3].map(k => this.storedVertex(triangles[source * 4 + k], layout, normal));
      this.flatCache?.set(source, vertices);

      return vertices;
    }

    return [1, 2, 3].map(k => {
      const offset = triangles[source * 4 + k];
      let vertex = this.vertexCache?.get(offset);

      if (!vertex) {
        vertex = this.storedVertex(offset, layout, null);
        this.vertexCache?.set(offset, vertex);
      }

      return vertex;
    });
  }

  /**
   * Side culling decided from clip coordinates alone, before reading attributes. With all three vertices in
   * front of the camera, clipping preserves orientation; otherwise the full path decides.
   */
  private culledBySide(source: number, lens: number[], winding: number, side: number): boolean {
    if (side !== 0 && side !== 1)
      return false;
    const { vertices: data, triangles } = this.geometry!, xy: number[] = [];

    for (let k = 1; k <= 3; k++) {
      const o = triangles[source * 4 + k], w = data[o + 3];
      if (!(w > 1e-10))
        return false;
      let x = data[o], y = data[o + 1];

      if (lens[0] || lens[1]) {
        x += this.projection[0] * lens[0] * (w / this.settings.dofFocusDistance - 1);
        y += this.projection[5] * lens[1] * (w / this.settings.dofFocusDistance - 1);
      }

      xy.push((x / w * .5 + .5) * this.width, (.5 - y / w * .5) * this.height);
    }

    const area = edge([xy[0], xy[1]], [xy[2], xy[3]], xy[4], xy[5]);
    if (Math.abs(area) < 1e-10)
      return false;
    const front = area * winding < 0;

    return side === 0 ? !front : front;
  }

  private project(source: number, lens: number[]): ScreenTriangle[] {
    const group = this.group(source);
    if (this.culledBySide(source, lens, group.winding, group.material.number("side", 0)))
      return [];

    const vertices = this.sourceVertices(source).map(v => {
      if (!lens[0] && !lens[1])
        return v;
      const clip = [...v.clip];
      clip[0] += this.projection[0] * lens[0] * (clip[3] / this.settings.dofFocusDistance - 1);
      clip[1] += this.projection[5] * lens[1] * (clip[3] / this.settings.dofFocusDistance - 1);

      return {
        ...v,
        clip,
      };
    });

    const polygon = clipPolygon(vertices), out: ScreenTriangle[] = [];

    for (let i = 1; i + 1 < polygon.length; i++) {
      const vs = [polygon[0], polygon[i], polygon[i + 1]], xy = vs.map(v =>
        [(v.clip[0] / v.clip[3] * .5 + .5) * this.width, (.5 - v.clip[1] / v.clip[3] * .5) * this.height]);

      let area = edge(xy[0], xy[1], xy[2][0], xy[2][1]);
      if (Math.abs(area) < 1e-10)
        continue;
      const front = area * group.winding < 0, side = group.material.number("side", 0);
      if ((side === 0 && !front) || (side === 1 && front))
        continue;

      if (area < 0) {
        [vs[1], vs[2]] = [vs[2], vs[1]];
        [xy[1], xy[2]] = [xy[2], xy[1]];
        area = -area;
      }

      out.push({

        vertices: vs,
        xy,
        area,
        front,
        source,

      });
    }

    return out;
  }

  private get orthographic(): boolean { return this.projection[15] === 1; }

  private viewRay(px: number, py: number) {
    const v = transform4(this.inverseProjection, [2 * px / this.width - 1, 1 - 2 * py / this.height, -1, 1]);
    const point = v.slice(0, 3).map(c => c / v[3]);
    const origin = this.orthographic ? transform4(this.cameraWorld, point).slice(0, 3) : this.cameraPosition;

    const direction = normalize(transform4(this.cameraWorld,
      [...(this.orthographic ? [0, 0, -1] : point), 0]).slice(0, 3));

    return { origin, direction };
  }

  private background(x: number, y: number): number[] {
    if (this.settings.backgroundMode === "transparent"
      || (this.settings.backgroundMode === "environment" && !this.environment.backgroundVisible))
      return [0, 0, 0, 0];

    const { direction } = this.viewRay(x, y);

    const color = this.settings.backgroundMode === "color" ? this.backgroundColor : scale(
      this.environment.sampleBackground(
        direction,
        this.settings.environmentRotation,
      ),
      this.settings.environmentIntensity,
    );

    return [...color, 1];
  }

  /** Per-triangle tangent frame: constant across the plane; only the fit to the interpolated normal is per fragment. */
  private tangentFrame(source: number) {
    if (source === this.tangentSource)
      return;
    const { vertices: data, triangles } = this.geometry!, layout = this.group(source).layout;
    let uvOffset = -1, cursor = VERTEX_HEADER;

    for (const [name, size] of layout.attributes) {
      if (name === "uv") {
        uvOffset = cursor;
        break;
      }

      cursor += size;
    }

    const world = (k: number) => {
      const o = triangles[source * 4 + k];

      return [data[o + 4], data[o + 5], data[o + 6]];
    };

    const uv = (k: number) => {
      const o = triangles[source * 4 + k] + uvOffset;

      return uvOffset < 0 ? ZERO3 : [data[o], data[o + 1]];
    };

    const a = world(1), dp1 = sub(world(2), a), dp2 = sub(world(3), a);
    const ua = uv(1), ub = uv(2), uc = uv(3);
    const du1 = ub[0] - ua[0], dv1 = ub[1] - ua[1], du2 = uc[0] - ua[0], dv2 = uc[1] - ua[1];
    const det = du1 * dv2 - du2 * dv1, faceNormal = normalize(cross(dp1, dp2));
    this.tangent = Math.abs(det) > 1e-10
      ? normalize(scale(sub(scale(dp1, dv2), scale(dp2, dv1)), 1 / det))
      : normalize(cross(Math.abs(faceNormal[1]) < .99 ? [0, 1, 0] : [1, 0, 0], faceNormal));
    this.bitangentSign = det < 0 ? -1 : 1;
    this.tangentSource = source;
  }

  private surface(
    triangle: ScreenTriangle,
    weights: number[],
    x: number,
    y: number,
    depth: number,
    lens: number[],
    total: number): Surface {
    const [a, b, c] = triangle.vertices, [w0, w1, w2] = weights;
    const attributes: Record<string, number[]> = {};

    for (const name in a.attributes) {
      const va = a.attributes[name], vb = b.attributes[name], vc = c.attributes[name];
      const out = new Array<number>(va.length);
      for (let i = 0; i < va.length; i++)
        out[i] = va[i] * w0 + vb[i] * w1 + vc[i] * w2;
      attributes[name] = out;
    }

    const varyings: Record<string, Value> = {};
    for (const name in a.varyings)
      varyings[name] = interpolate3(a.varyings[name], b.varyings[name], c.varyings[name], w0, w1, w2);

    const position = [
      a.world[0] * w0 + b.world[0] * w1 + c.world[0] * w2,
      a.world[1] * w0 + b.world[1] * w1 + c.world[1] * w2,
      a.world[2] * w0 + b.world[2] * w1 + c.world[2] * w2,
    ];

    const normal = normalize([
      a.normal[0] * w0 + b.normal[0] * w1 + c.normal[0] * w2,
      a.normal[1] * w0 + b.normal[1] * w1 + c.normal[1] * w2,
      a.normal[2] * w0 + b.normal[2] * w1 + c.normal[2] * w2,
    ]);

    return this.finishSurface(triangle.source, triangle.front, position, normal, attributes, varyings,
      x, y, depth, lens, total);
  }

  /** Extrapolate the same triangle into globally aligned GL quads, including uncovered helper pixels. */
  private attachDerivativeQuad(surface: Surface, triangle: ScreenTriangle, lens: number[], polygonOffset = 0) {
    const x = surface.fragCoord[0], y = surface.fragCoord[1];
    const ix = Math.floor(x), iy = Math.floor(y), lane = (ix & 1) + 2 * (iy & 1);
    const qx = ix - (ix & 1) + (x - ix), qy = iy - (iy & 1) + (y - iy);
    let helpers: Surface[] | undefined;
    surface.quad = { lane, surfaces: () => helpers ??= [0, 1, 2, 3].map(i => {
      if (i === lane) return surface;
      const px = qx + (i & 1), py = this.height - (qy + (i >> 1));
      const [a, b, c] = triangle.xy, vs = triangle.vertices;

      const bary = [edge(b, c, px, py), edge(c, a, px, py), edge(a, b, px, py)]
        .map(v => v / triangle.area);

      const perspective = bary.map((v, j) => v / vs[j].clip[3]);
      const total = perspective[0] + perspective[1] + perspective[2];
      const depth = bary.reduce((z, v, j) => z + v * (vs[j].clip[2] / vs[j].clip[3] * .5 + .5), polygonOffset);

      return this.surface(triangle, perspective.map(v => v / total), px, py, depth, lens, total);
    }) };
  }

  /** Same interpolation as `surface`, straight from the prepared geometry: ray hits do not build vertices. */
  private storedSurface(source: number, front: boolean, w0: number, w1: number, w2: number): Surface {
    const { vertices: data, triangles, faceNormals } = this.geometry!,
      group = this.group(source), layout = group.layout;

    const a = triangles[source * 4 + 1], b = triangles[source * 4 + 2], c = triangles[source * 4 + 3];
    const attributes: Record<string, number[]> = {};
    let cursor = VERTEX_HEADER;

    // A plain material reads no attributes: a ray hit only needs position and normal.
    for (const [name, size] of group.material.needsAttributes ? layout.attributes : []) {
      const out = new Array<number>(size);

      for (let i = 0; i < size; i++) {
        const k = cursor + i;
        out[i] = data[a + k] * w0 + data[b + k] * w1 + data[c + k] * w2;
      }

      attributes[name] = out;
      cursor += size;
    }

    const varyings: Record<string, Value> = {}, position = [cursor];
    for (const [name, shape] of group.material.needsAttributes ? layout.varyings : [])
      varyings[name] = interpolateStored(data, a, b, c, shape, w0, w1, w2, position);

    const point = [
      data[a + 4] * w0 + data[b + 4] * w1 + data[c + 4] * w2,
      data[a + 5] * w0 + data[b + 5] * w1 + data[c + 5] * w2,
      data[a + 6] * w0 + data[b + 6] * w1 + data[c + 6] * w2,
    ];

    const n = source * 3, flat = this.hasFlag(source, FLAT_NORMAL);

    const normal = flat
      ? normalize([
        faceNormals[n] * w0 + faceNormals[n] * w1 + faceNormals[n] * w2,
        faceNormals[n + 1] * w0 + faceNormals[n + 1] * w1 + faceNormals[n + 1] * w2,
        faceNormals[n + 2] * w0 + faceNormals[n + 2] * w1 + faceNormals[n + 2] * w2,
      ])
      : normalize([
        data[a + 7] * w0 + data[b + 7] * w1 + data[c + 7] * w2,
        data[a + 8] * w0 + data[b + 8] * w1 + data[c + 8] * w2,
        data[a + 9] * w0 + data[b + 9] * w1 + data[c + 9] * w2,
      ]);

    return this.finishSurface(source, front, point, normal, attributes, varyings, 0, 0, 0, [0, 0], 1);
  }

  private finishSurface(
    source: number,
    front: boolean,
    position: number[],
    normal: number[],
    attributes: Record<string, number[]>,
    varyings: Record<string, Value>,
    x: number,
    y: number,
    depth: number,
    lens: number[],
    total: number): Surface {
    const group = this.group(source), material = group.material;
    let tangent = ZERO3, bitangent = ZERO3;

    if (material.needsTangents) {
      this.tangentFrame(source);
      tangent = normalize(sub(this.tangent, scale(normal, dot(this.tangent, normal))));
      bitangent = scale(cross(normal, tangent), this.bitangentSign);
    }

    let cx = this.cameraPosition[0], cy = this.cameraPosition[1], cz = this.cameraPosition[2];

    if (lens[0] || lens[1]) {
      const lensWorld = transform4(this.cameraWorld, [lens[0], lens[1], 0, 0]);
      cx += lensWorld[0]; cy += lensWorld[1]; cz += lensWorld[2];
    }

    const view = this.orthographic ? normalize(this.cameraWorld.slice(8, 11))
      : normalize([cx - position[0], cy - position[1], cz - position[2]]);

    // Only shaders read per-fragment uniforms; built-in materials use the group's shared object.
    const uniforms = material.needsUniforms
      ? { ...group.uniforms, vViewPosition: scale(transform4(this.view, position).slice(0, 3), -1) }
      : group.uniforms;

    return {
      position,
      normal,
      tangent,
      bitangent,
      view,
      attributes,
      varyings,
      frontFacing: front,
      depth,
      fragCoord: [x, this.height - y, depth, total],
      uniforms,
    };
  }

  /** World-space barycentrics also interpolate UVs, vertex colors, shader varyings and normal maps. */
  private raySurface(hit: RayHit, direction: number[]): Surface {
    const p = this.geometry!.positions, o = hit.index * 9, p0 = [p[o], p[o + 1], p[o + 2]];
    const face = cross(sub([p[o + 3], p[o + 4], p[o + 5]], p0), sub([p[o + 6], p[o + 7], p[o + 8]], p0));
    const front = dot(face, direction) * this.group(hit.index).winding * this.cameraHandedness < 0;
    const surface = this.storedSurface(hit.index, front, 1 - hit.u - hit.v, hit.u, hit.v);
    surface.view = scale(direction, -1);
    // Camera-dependent shader coordinates remain a compatibility approximation for secondary hits.
    const clip = transform4(this.projection, transform4(this.view, surface.position));

    if (Math.abs(clip[3]) > 1e-10) {
      surface.depth = clip[2] / clip[3] * .5 + .5;
      surface.fragCoord = [(clip[0] / clip[3] * .5 + .5) * this.width,
        (clip[1] / clip[3] * .5 + .5) * this.height, surface.depth, 1 / clip[3]];
    }

    if (this.group(hit.index).material.needsFragmentQuad) {
      // Secondary hits use the camera projection, like their existing gl_FragCoord compatibility approximation.
      const vertices = this.sourceVertices(hit.index);

      const xy = vertices.map(v => [(v.clip[0] / v.clip[3] * .5 + .5) * this.width,
        (.5 - v.clip[1] / v.clip[3] * .5) * this.height]);

      const area = edge(xy[0], xy[1], xy[2][0], xy[2][1]);
      if (Number.isFinite(area) && Math.abs(area) > 1e-10)
        this.attachDerivativeQuad(surface, { vertices, xy, area, front, source: hit.index }, [0, 0]);
    }

    return surface;
  }

  private shadowAccept = (index: number, u: number, v: number): boolean => {
    if (this.hasFlag(index, SHADOW_OPAQUE))
      return true;
    const group = this.group(index), m = group.material;
    // Only texture coordinates and vertex color affect alpha.
    const { vertices: data, triangles } = this.geometry!;
    const a = triangles[index * 4 + 1], b = triangles[index * 4 + 2], c = triangles[index * 4 + 3], w0 = 1 - u - v;
    const attributes: Record<string, number[]> = {};
    let cursor = VERTEX_HEADER;

    for (const [name, size] of group.layout.attributes) {
      if (name.startsWith("uv") || name === "color") {
        const out = new Array<number>(size);

        for (let i = 0; i < size; i++) {
          const k = cursor + i;
          out[i] = data[a + k] * w0 + data[b + k] * u + data[c + k] * v;
        }

        attributes[name] = out;
      }

      cursor += size;
    }

    let alpha = m.alpha(attributes);
    if (m.flag("vertexColors") && attributes.color)
      alpha *= attributes.color[3] ?? 1;

    return alpha >= shadowThreshold(m);
  };

  /** Ray origin, offset slightly off the surface to avoid hitting its own triangle. */
  private rayOrigin(position: number[], normal: number[]) {
    const bias = Math.max(1e-5, Math.max(Math.abs(position[0]), Math.abs(position[1]), Math.abs(position[2])) * 1e-6);

    return [position[0] + normal[0] * bias, position[1] + normal[1] * bias, position[2] + normal[2] * bias];
  }

  private visibility(source: number, position: number[], normal: number[], direction: number[], distance: number,
    seed: number, soft = true, ignoreSelf = false): number {
    if (!this.settings.shadows || !this.hasFlag(source, RECEIVE_SHADOW) || !this.bvh)
      return 1;
    const origin = this.rayOrigin(position, normal);
    const softness = soft ? (this.settings.shadowSoftness ?? 0) * Math.PI / 180 : 0;
    const triangles = this.geometry!.triangles, own = triangles[source * 4];

    const accept = ignoreSelf
      ? (index: number, u: number, v: number) => triangles[index * 4] !== own && this.shadowAccept(index, u, v)
      : this.shadowAccept;

    if (softness <= 0)
      return this.bvh.occluded(origin, direction, distance, source, accept) ? 0 : 1;
    // Soft shadow: the light becomes a disk of this angular diameter; the fraction of unblocked rays is the penumbra.
    const { t, b } = frame(direction), spread = Math.tan(softness / 2);
    let lit = 0;

    for (let i = 0; i < SHADOW_SAMPLES; i++) {
      const u = (halton(i + 1, 2) + seed) % 1, v = (halton(i + 1, 3) + seed * .618034) % 1;
      const r = Math.sqrt(u) * spread, a = v * 2 * Math.PI;
      const cu = r * Math.cos(a), sv = r * Math.sin(a), d = this.rayDirection;
      const x = direction[0] + (t[0] * cu + b[0] * sv);
      const y = direction[1] + (t[1] * cu + b[1] * sv);
      const z = direction[2] + (t[2] * cu + b[2] * sv);
      const inverse = 1 / (Math.sqrt(x * x + y * y + z * z) || 1);
      d[0] = x * inverse; d[1] = y * inverse; d[2] = z * inverse;
      if (!this.bvh.occluded(origin, d, distance, source, accept))
        lit++;
    }

    return lit / SHADOW_SAMPLES;
  }

  /** View depth of a world-space point (positive in front of the camera). */
  private viewDepth(p: number[]) {
    const v = this.view;

    return -(v[2] * p[0] + v[6] * p[1] + v[10] * p[2] + v[14]);
  }

  /** World-space size of a pixel at that depth: scales the irradiance cache radii. */
  private pixelFootprint(depth: number) {
    return (this.orthographic ? 1 : Math.max(depth, 1e-3)) * 2 / Math.abs(this.projection[5]) / this.height;
  }

  /** Computes indirect light at a point and stores the record for its neighbors in the bucket. */
  private cacheRecord(source: number, position: number[], normal: number[], seed: number) {
    const estimate = this.gi!.estimate(source, position, normal, seed);
    const spacing = this.pixelFootprint(this.viewDepth(position)) * (this.settings.giCacheSpacing ?? 4);
    const radius = clamp(estimate.radius * CACHE_ACCURACY, spacing * CACHE_MIN_SPACING, spacing * CACHE_MAX_SPACING);
    this.giCache!.add({ position, normal, color: estimate.color, variance: estimate.variance, radius });

    return estimate;
  }

  /** Indirect light of a fragment: from the cache when some record reaches the point, otherwise computed and stored. */
  private indirect(source: number, position: number[], normal: number[], seed: number): number[] {
    if (!this.giCache) {
      const estimate = this.gi!.estimate(source, position, normal, seed);
      this.lastVariance = estimate.variance;

      return estimate.color;
    }

    const cached = this.giGrid?.lookupAt(this.fragmentX, this.fragmentY, position, normal, 0, LOOKUP_SMOOTHING)
      ?? this.giCache.lookup(position, normal, LOOKUP_SMOOTHING);

    if (cached) {
      this.lastVariance = cached.variance;

      return cached.color;
    }

    const estimate = this.cacheRecord(source, position, normal, seed);
    this.lastVariance = estimate.variance;

    return estimate.color;
  }

  /** Ray-traced ambient occlusion: fraction of the (cosine-weighted) hemisphere blocked within aoDistance. */
  private occlusion(source: number, position: number[], normal: number[], seed: number): number {
    const intensity = this.settings.aoIntensity ?? 1, distance = this.settings.aoDistance ?? 1.5;
    if (this.gi || !this.settings.ambientOcclusion || !this.bvh || intensity <= 0 || distance <= 0)
      return 1;
    const origin = this.rayOrigin(position, normal), { t, b } = frame(normal);
    let hits = 0;

    for (let i = 0; i < AO_SAMPLES; i++) {
      const u = (halton(i + 1, 2) + seed) % 1, v = (halton(i + 1, 3) + seed * .618034) % 1;
      const r = Math.sqrt(u), a = v * 2 * Math.PI;
      const cu = r * Math.cos(a), sv = r * Math.sin(a), up = Math.sqrt(1 - r * r), d = this.rayDirection;
      d[0] = (t[0] * cu + b[0] * sv) + normal[0] * up;
      d[1] = (t[1] * cu + b[1] * sv) + normal[1] * up;
      d[2] = (t[2] * cu + b[2] * sv) + normal[2] * up;
      if (this.bvh.occluded(origin, d, distance, source, this.shadowAccept))
        hits++;
    }

    return Math.max(0, 1 - intensity * hits / AO_SAMPLES);
  }

  /**
   * One completed bucket is returned at full sample quality. `progress` receives each pixel of the final image as
   * soon as it is resolved, in row order (with depth of field, once per lens pass, averaging the passes so far).
   */
  *renderBucket(
    bucket: Bucket,
    progress?: (pixels: Uint8ClampedArray, pixel: number) => void): Generator<void, BucketResult> {
    this.vertexCache = new Map();
    this.flatCache = new Map();
    // With the full-image grid the bucket only keeps fallback records; without it, it builds its own grid.
    this.giCache = this.usesCache ? new IrradianceCache() : null;
    this.cachePlaced = this.giGrid !== null;
    const count = bucket.width * bucket.height, samples = this.settings.maxSamples;
    const sum = new Float32Array(count * 4);

    // Denoiser accumulators: summed modulation, coverage-weighted indirect light and its weight.
    const aov = this.aov && !this.target.linear ? {
      modulation: new Float32Array(count * 3), indirect: new Float32Array(count * 3), weight: new Float32Array(count),
      normal: new Float32Array(count * 3), depth: new Float32Array(count), variance: new Float32Array(count),
    } : null;

    const targetDepth = this.target.linear ? new Float32Array(count).fill(1) : undefined;
    // Background once per pixel, at the center: the environment is smooth and needs no per-sample antialiasing.
    const background = new Float32Array(count * 4);
    for (let y = 0; y < bucket.height; y++)
      for (let x = 0; x < bucket.width; x++)
        background.set(this.background(bucket.x + x + .5, bucket.y + y + .5), (y * bucket.width + x) * 4);
    const jitter = (sample: number) => samples === 1 ? [.5, .5] : [halton(sample + 1, 2), halton(sample + 1, 3)];
    const pixels = new Uint8ClampedArray(count * 4);
    // Vignette and grain are camera effects: only on the final image, never on linear captures.
    const vignette = this.target.linear ? 0 : clamp(this.settings.vignette ?? 0);
    const grain = this.target.linear ? 0 : clamp(this.settings.grain ?? 0);
    // Linear captures resolve inside their own accumulator, so only the final image is shown while rendering.
    const live = this.target.linear ? undefined : progress;

    const resolved = (done: number) => live ? (pixel: number) => {
      this.resolvePixel(bucket, pixel, sum, done, pixels, vignette, grain);
      live(pixels, pixel);
    } : undefined;

    // Grid built right here (no separate prepass): each record shows as a dot until shading covers it.
    const onRecord = live && this.giCache && !this.cachePlaced ? this.recordPreview(bucket, pixels, live) : undefined;

    if (this.lensRadius > 0) {
      // With depth of field each sample sees the scene from a different lens point: one pass per sample.
      for (let sample = 0; sample < samples; sample++) {
        let radius = Math.sqrt(halton(sample + 1, 5)) * this.lensRadius;
        const angle = halton(sample + 1, 7) * 2 * Math.PI, blades = this.settings.bokehBlades;
        if (blades >= 3)
          radius *= Math.cos(Math.PI / blades) / Math.cos((angle % (2 * Math.PI / blades)) - Math.PI / blades);
        yield* this.renderPass(bucket, [jitter(sample)], [radius * Math.cos(angle), radius * Math.sin(angle)],
          background, sum, targetDepth, resolved(sample + 1), aov, onRecord);
      }
    } else {
      // Without a lens there is a single projection: coverage per sample, shading once per pixel per triangle.
      yield* this.renderPass(bucket, Array.from({ length: samples }, (_, sample) => jitter(sample)), [0, 0],
        background, sum, targetDepth, resolved(samples), aov, onRecord);
    }

    this.vertexCache = null;
    this.flatCache = null;
    this.giCache = null;
    if (aov)
      this.storeAov(bucket, sum, aov);

    // With progressive output the last pass has already resolved every pixel with all samples.
    if (!live) {
      for (let i = 0; i < count; i++) {
        if (!this.target.linear) {
          this.resolvePixel(bucket, i, sum, samples, pixels, vignette, grain);
          continue;
        }

        const alpha = sum[i * 4 + 3] / samples;
        const rgb = [0, 1, 2].map(c => alpha ? sum[i * 4 + c] / samples / alpha : 0);
        sum.set([...rgb, alpha], i * 4);
        pixels.set([...encodeColor(rgb, 1, "linear", false), Math.round(clamp(alpha) * 255)], i * 4);
      }
    }

    return {
      bucket,
      pixels,
      linear: this.target.linear ? sum : undefined,
      depth: targetDepth,
    };
  }

  /** The bucket's triangles projected for this lens and, per pixel, which ones cover any sample. */
  private *preparePass(bucket: Bucket, jitters: number[][], lens: number[]):
  Generator<void, { passTriangles: PassTriangle[]; candidates: (number[] | undefined)[] }> {
    const count = bucket.width * bucket.height;
    const projected: ScreenTriangle[] = [];

    if (this.geometry) {
      const { binOffsets, binTriangles } = this.geometry;
      for (let k = binOffsets[bucket.index]; k < binOffsets[bucket.index + 1]; k++)
        projected.push(...this.project(binTriangles[k], lens));
    }

    projected.sort((a, b) => this.group(a.source).renderOrder - this.group(b.source).renderOrder);
    const passTriangles = projected.map(triangle => passTriangle(triangle, this.group(triangle.source).material));
    const candidates: (number[] | undefined)[] = new Array(count);

    for (let t = 0; t < passTriangles.length; t++) {
      const [p0, p1, p2] = passTriangles[t].triangle.xy;
      const minX = Math.max(bucket.x, Math.floor(Math.min(p0[0], p1[0], p2[0])));
      const maxX = Math.min(bucket.x + bucket.width - 1, Math.floor(Math.max(p0[0], p1[0], p2[0])));
      const minY = Math.max(bucket.y, Math.floor(Math.min(p0[1], p1[1], p2[1])));
      const maxY = Math.min(bucket.y + bucket.height - 1, Math.floor(Math.max(p0[1], p1[1], p2[1])));

      for (let y = minY; y <= maxY; y++) {
        for (let x = minX; x <= maxX; x++) {
          if (coversPixel(passTriangles[t], x, y, jitters))
            (candidates[(y - bucket.y) * bucket.width + x - bucket.x] ??= []).push(t);
        }

        yield;
      }
    }

    return { passTriangles, candidates };
  }

  /** Record preview: a small square in the indirect light color at its pixel, sent through the progress channel. */
  private recordPreview(
    bucket: Bucket, pixels: Uint8ClampedArray, live: (pixels: Uint8ClampedArray, pixel: number) => void) {
    const dot = Math.max(1, Math.floor((this.settings.giCacheSpacing ?? 4) / 2)), preview = new Float32Array(4);
    const vignette = clamp(this.settings.vignette ?? 0), grain = clamp(this.settings.grain ?? 0);

    return (pixel: number, rgb: number[]) => {
      preview[0] = rgb[0]; preview[1] = rgb[1]; preview[2] = rgb[2]; preview[3] = 1;
      const cx = pixel % bucket.width, cy = Math.floor(pixel / bucket.width);

      for (let dy = 0; dy < dot; dy++)
        for (let dx = 0; dx < dot; dx++) {
          const x = cx + dx, y = cy + dy;
          if (x >= bucket.width || y >= bucket.height)
            continue;
          const i = y * bucket.width + x;
          this.resolveSample(bucket.x + x + .5, bucket.y + y + .5, preview, 0, 1, pixels, i * 4, vignette, grain);
          live(pixels, i);
        }
    };
  }

  /**
   * Irradiance cache records on a pixel grid over the bucket: at each point, the nearest opaque surface among the
   * materials that query indirect light gets a full computation, which its neighbors interpolate.
   */
  private *placeCacheRecords(
    bucket: Bucket,
    candidates: (number[] | undefined)[],
    passTriangles: PassTriangle[],
    lens: number[],
    clipPlane: number[] | undefined,
    onRecord?: (pixel: number, color: number[]) => void): Generator<void> {
    const spacing = Math.max(1, Math.round(this.settings.giCacheSpacing ?? 4)), count = bucket.width * bucket.height;
    let lastYield = performance.now();

    for (let pixel = 0; pixel < count; pixel++) {
      const x = bucket.x + pixel % bucket.width, y = bucket.y + Math.floor(pixel / bucket.width);
      const list = candidates[pixel];
      if (x % spacing !== 0 || y % spacing !== 0 || !list)
        continue;
      const px = x + .5, py = y + .5;
      // Opaque surfaces covering the center, nearest to farthest: the first one that passes the alpha test
      // gets the record, so cutout holes get no point and the surface behind shows through.
      const covering: { t: number; z: number; b0: number; b1: number; b2: number }[] = [];

      for (const t of list) {
        const pass = passTriangles[t];
        if (pass.translucent || !pass.depthTest || !pass.colorWrite
          || !this.group(pass.triangle.source).material.diffuseGi)
          continue;
        const [p0, p1, p2] = pass.triangle.xy;
        const e0 = edge(p1, p2, px, py), e1 = edge(p2, p0, px, py), e2 = edge(p0, p1, px, py);
        if (e0 < 0 || e1 < 0 || e2 < 0)
          continue;
        const b0 = e0 * pass.invArea, b1 = e1 * pass.invArea, b2 = e2 * pass.invArea;
        covering.push({ t, z: b0 * pass.z0 + b1 * pass.z1 + b2 * pass.z2 + pass.polygonOffset, b0, b1, b2 });
      }

      covering.sort((a, b) => a.z - b.z);

      for (const { t, z, b0, b1, b2 } of covering) {
        const pass = passTriangles[t], triangle = pass.triangle, material = this.group(triangle.source).material;
        const i0 = b0 * pass.invW0, i1 = b1 * pass.invW1, i2 = b2 * pass.invW2, total = i0 + i1 + i2;
        const surface = this.surface(triangle, [i0 / total, i1 / total, i2 / total], px, py, z, lens, total);
        if (clipPlane && dot(clipPlane, surface.position) + clipPlane[3] < 0)
          continue;
        const alphaTest = material.number("alphaTest", 0);

        if (alphaTest > 0) {
          // Same mip level as the fragment pass, so records land on the same cutout holes as the pixels.
          if (material.needsFragmentQuad) this.attachDerivativeQuad(surface, triangle, lens, pass.polygonOffset);
          let alpha = material.alpha(surface.attributes, material.uvGradients(surface));
          if (material.flag("vertexColors") && surface.attributes.color)
            alpha *= surface.attributes.color[3] ?? 1;
          if (alpha < alphaTest)
            continue;
        }

        const normal = scale(normalize(surface.normal), triangle.front ? 1 : -1);
        const estimate = this.cacheRecord(triangle.source, surface.position, normal, hash(px, py));
        // Record preview: indirect light tinted by the base color, so the dot resembles the surface.
        onRecord?.(pixel, mul(estimate.color, material.vector("color", [1, 1, 1])));
        break;
      }

      if (performance.now() - lastYield > 8) {
        yield;
        lastYield = performance.now();
      }
    }
  }

  /** Copies the bucket accumulators into the denoiser's full-image buffers. */
  private storeAov(bucket: Bucket, sum: Float32Array, local: BucketAov) {
    const frame = this.aov!;

    for (let y = 0; y < bucket.height; y++)
      for (let x = 0; x < bucket.width; x++) {
        const i = y * bucket.width + x, g = (bucket.y + y) * frame.width + bucket.x + x, w = local.weight[i];
        frame.color.set(sum.subarray(i * 4, i * 4 + 4), g * 4);
        frame.modulation.set(local.modulation.subarray(i * 3, i * 3 + 3), g * 3);
        for (let c = 0; c < 3; c++)
          frame.indirect[g * 3 + c] = w > 0 ? local.indirect[i * 3 + c] / w : 0;
        frame.normal.set(local.normal.subarray(i * 3, i * 3 + 3), g * 3);
        frame.depth[g] = local.depth[i];
        frame.variance[g] = local.variance[i];
      }
  }

  /**
   * Recomposes the full image with the filtered indirect light: each pixel receives the difference between the
   * filtered and the original indirect light, times the modulation, and goes through the same resolve as buckets.
   * `onRows` receives bands of finished rows; the last call completes the image.
   */
  *resolveDenoised(
    filtered: Float32Array,
    onRows: (row: number, pixels: Uint8ClampedArray) => void): Generator<void> {
    const frame = this.aov;
    if (!frame)
      throw new Error("CPU renderer: no denoise buffers were attached.");
    const samples = this.settings.maxSamples, sum = new Float32Array(4);
    const vignette = clamp(this.settings.vignette ?? 0), grain = clamp(this.settings.grain ?? 0);
    let rows = new Uint8ClampedArray(this.width * 4 * Math.min(this.height, 16)), firstRow = 0, filled = 0;
    let lastYield = performance.now();

    for (let y = 0; y < this.height; y++) {
      for (let x = 0; x < this.width; x++) {
        const g = y * this.width + x;
        for (let c = 0; c < 3; c++)
          sum[c] = frame.color[g * 4 + c]
            + frame.modulation[g * 3 + c] * (filtered[g * 3 + c] - frame.indirect[g * 3 + c]);
        sum[3] = frame.color[g * 4 + 3];
        this.resolveSample(x + .5, y + .5, sum, 0, samples, rows, (g - firstRow * this.width) * 4, vignette, grain);
      }

      filled++;

      if (filled * this.width * 4 >= rows.length || y === this.height - 1) {
        onRows(firstRow, rows.subarray(0, filled * this.width * 4));
        firstRow = y + 1; filled = 0;
        rows = new Uint8ClampedArray(this.width * 4 * Math.min(this.height - firstRow, 16));
      }

      if (performance.now() - lastYield > 8) {
        yield;
        lastYield = performance.now();
      }
    }
  }

  /** Converts a pixel's linear accumulator into the displayed color, using the `samples` taken so far. */
  private resolvePixel(
    bucket: Bucket,
    i: number,
    sum: Float32Array,
    samples: number,
    pixels: Uint8ClampedArray,
    vignette: number,
    grain: number) {
    this.resolveSample(bucket.x + (i % bucket.width) + .5, bucket.y + Math.floor(i / bucket.width) + .5,
      sum, i * 4, samples, pixels, i * 4, vignette, grain);
  }

  /** Resolves the accumulator read at `offset` into the pixel written at `out`, at image position `px`, `py`. */
  private resolveSample(
    px: number,
    py: number,
    sum: Float32Array,
    offset: number,
    samples: number,
    pixels: Uint8ClampedArray,
    out: number,
    vignette: number,
    grain: number) {
    const alpha = sum[offset + 3] / samples;
    // With volumetrics, reproduce the composer's premultiplied HDR buffer through tone mapping.
    // Unpremultiplying here changes the tone curve and still lets Canvas 2D attenuate the glow again.
    const inv = this.volumetric.active ? 1 / samples : alpha ? 1 / (samples * alpha) : 0;
    let r = sum[offset] * inv, g = sum[offset + 1] * inv, b = sum[offset + 2] * inv;

    if (vignette > 0) {
      const u = px / this.width * 2 - 1, v = py / this.height * 2 - 1;
      const k = 1 - vignette * smoothstep(.25, 2, u * u + v * v);
      r *= k; g *= k; b *= k;
    }

    // The accumulator is linear; tone mapping applies only to the final image, as in GPU post-processing.
    const mapped = toneMap3(r, g, b, this.settings.exposure, this.settings.tonemapping);
    let noise = 0;
    if (grain > 0)
      noise = (hash(px, py) - .5) * grain * .2;
    const encoded = mapped.map(v => clamp(linearToSrgb(clamp(v + noise))));
    // WebGL's canvas consumes premultiplied sRGB; ImageData consumes straight sRGB.
    // ADD can produce RGB > alpha, which a straight-alpha PNG cannot represent. Raise coverage
    // just enough to retain every emitted channel, then unpremultiply in sRGB (not linear light).
    const outputAlpha = this.volumetric.active ? Math.max(clamp(alpha), ...encoded) : clamp(alpha);
    const unpremultiply = this.volumetric.active && outputAlpha > 0 ? 1 / outputAlpha : 1;
    for (let c = 0; c < 3; c++) pixels[out + c] = Math.round(encoded[c] * unpremultiply * 255);
    pixels[out + 3] = Math.round(outputAlpha * 255);
  }

  /**
   * A pass rasterizes all samples of a single projection; the premultiplied result is added to `sum`.
   * It first records, per pixel, the triangles that cover it; then shades pixel by pixel in row order,
   * replaying each pixel's triangles in draw order. A pixel's depth and transparency depend only on
   * that sequence, so the image matches the per-triangle loop, and each pixel comes out finished.
   */
  private *renderPass(
    bucket: Bucket,
    jitters: number[][],
    lens: number[],
    background: Float32Array,
    sum: Float32Array,
    targetDepth: Float32Array | undefined,
    onPixel?: (pixel: number) => void,
    aov: BucketAov | null = null,
    onRecord?: (pixel: number, color: number[]) => void): Generator<void> {
    const count = bucket.width * bucket.height, S = jitters.length;
    const { passTriangles, candidates } = yield* this.preparePass(bucket, jitters, lens);
    // Each pixel finishes before the next: per-sample color and depth only need room for one pixel.
    const color = new Float32Array(S * 4), depth = new Float64Array(S);
    let current = -1, seed = 0;

    const ctx: LightingContext = {
      lights: this.scene.lights,
      environment: this.environment,
      settings: this.settings,
      visibility: (p, n, d, dist, ignoreSelf) => this.visibility(current, p, n, d, dist, seed, true, ignoreSelf),
      occlusion: (p, n) => this.occlusion(current, p, n, seed),
      indirect: this.gi ? (p, n) => this.indirect(current, p, n, seed) : undefined,
      transmitted: (exit, blur) => {
        // The nested shading borrows the per-fragment state; the transmissive fragment gets it back afterwards.
        const source = current, fx = this.fragmentX, fy = this.fragmentY, variance = this.lastVariance;

        const light = this.transmittedLight(exit, blur, seed, (index, surface, px, py) => {
          current = index;
          this.fragmentX = px; this.fragmentY = py;

          return this.group(index).material.shade(surface, behind);
        });

        current = source;
        this.fragmentX = fx; this.fragmentY = fy;
        this.lastVariance = variance;

        return light;
      },
    };

    // What glass sees behind it, as in Three's transmission pass: lit and shadowed, without ambient occlusion and with
    // hard shadows. The lookup blurs it anyway, and the extra rays would only add noise and cost to every tap.
    const behind: LightingContext = {
      ...ctx,
      visibility: (p, n, d, dist, ignoreSelf) => this.visibility(current, p, n, d, dist, seed, false, ignoreSelf),
      occlusion: () => 1,
      transmitted: undefined,
    };

    let lastYield = performance.now();
    const covered = new Float64Array(S), clipPlane = this.target.clipPlane;

    if (this.giCache && !this.cachePlaced) {
      yield* this.placeCacheRecords(bucket, candidates, passTriangles, lens, clipPlane, onRecord);
      this.cachePlaced = true;
    }

    // Per sample: modulation, indirect light and coverage of the visible opaque fragment, for the denoiser.
    const modS = new Float32Array(S * 3), indS = new Float32Array(S * 3), weightS = new Float32Array(S);

    for (let pixel = 0; pixel < count; pixel++) {
      const x = bucket.x + pixel % bucket.width, y = bucket.y + Math.floor(pixel / bucket.width);

      const b0 = background[pixel * 4], b1 = background[pixel * 4 + 1],
        b2 = background[pixel * 4 + 2], b3 = background[pixel * 4 + 3];

      for (let s = 0; s < S; s++) {
        color[s * 4] = b0; color[s * 4 + 1] = b1; color[s * 4 + 2] = b2; color[s * 4 + 3] = b3;
        depth[s] = Infinity;
      }

      if (aov) {
        modS.fill(0); indS.fill(0); weightS.fill(0);
      }

      let entries: TransparentEntry[] | undefined;
      const list = candidates[pixel];

      if (list) for (let li = 0; li < list.length; li++) {
        const t = list[li];
        const pass = passTriangles[t], triangle = pass.triangle;
        const group = this.group(triangle.source), material = group.material;
        current = triangle.source;
        const [p0, p1, p2] = triangle.xy;
        let mask = 0, first = -1, fx = 0, fy = 0, fb0 = 0, fb1 = 0, fb2 = 0;

        for (let s = 0; s < S; s++) {
          const px = x + jitters[s][0], py = y + jitters[s][1];
          const e0 = edge(p1, p2, px, py), e1 = edge(p2, p0, px, py), e2 = edge(p0, p1, px, py);
          if (e0 < 0 || e1 < 0 || e2 < 0
            || (e0 === 0 && !pass.tl0) || (e1 === 0 && !pass.tl1) || (e2 === 0 && !pass.tl2))
            continue;
          if (pass.wireframe && Math.min(e0 / pass.len0, e1 / pass.len1, e2 / pass.len2) > pass.lineWidth)
            continue;
          const b0 = e0 * pass.invArea, b1 = e1 * pass.invArea, b2 = e2 * pass.invArea;
          const z = b0 * pass.z0 + b1 * pass.z1 + b2 * pass.z2 + pass.polygonOffset;
          if (pass.earlyZ && !depthPass(pass.depthFunc, z, depth[s]))
            continue;
          covered[s] = z;
          mask |= 1 << s;

          if (first < 0) {
            first = s; fx = px; fy = py; fb0 = b0; fb1 = b1; fb2 = b2;
          }
        }

        if (!mask)
          continue;
        // Perspective-correct weights at the first covered sample: one shading serves the rest.
        const i0 = fb0 * pass.invW0, i1 = fb1 * pass.invW1, i2 = fb2 * pass.invW2, total = i0 + i1 + i2;

        const surface = this.surface(triangle, [i0 / total, i1 / total, i2 / total], fx, fy, covered[first],
          lens, total);

        if (clipPlane && dot(clipPlane, surface.position) + clipPlane[3] < 0)
          continue;
        if (material.needsFragmentQuad) this.attachDerivativeQuad(surface, triangle, lens, pass.polygonOffset);
        seed = hash(fx, fy);
        this.fragmentX = fx; this.fragmentY = fy;
        this.lastVariance = 0;
        const fragment = material.shade(surface, ctx);
        const variance = this.lastVariance;

        // Expensive fragments (GI, water) must not hog the worker: yield control every few milliseconds.
        if (performance.now() - lastYield > 8) {
          yield;
          lastYield = performance.now();
        }

        if (!fragment)
          continue;
        if (!fragment.color.every(Number.isFinite) || !Number.isFinite(fragment.alpha)
          || !Number.isFinite(fragment.depth))
          throw new Error(`${material.data.name}: fragment contains non-finite values.`);

        this.applyFog(fragment, surface.position, material);

        // gl_FragDepth written by the shader applies to all samples; otherwise each sample uses its own z.
        const shaderDepth = fragment.depth !== covered[first] ? clamp(fragment.depth) : -1;

        if (pass.translucent) {
          (entries ??= []).push({
            fragment,
            material,
            order: group.renderOrder,
            mask,
            depths: shaderDepth >= 0 ? null : Float64Array.from(covered),
            depth: shaderDepth,
          });
          continue;
        }

        const info = aov && fragment.indirect && fragment.modulation ? fragment : null;
        const fragmentNormal = info ? scale(normalize(surface.normal), triangle.front ? 1 : -1) : ZERO3;

        for (let s = 0; s < S; s++) {
          if (!(mask & (1 << s)))
            continue;
          const z = shaderDepth >= 0 ? shaderDepth : covered[s];
          if (pass.depthTest && !depthPass(pass.depthFunc, z, depth[s]))
            continue;
          if (pass.depthWrite)
            depth[s] = z;

          if (pass.colorWrite) {
            const offset = s * 4, alpha = fragment.alpha;
            color[offset] = fragment.color[0] * alpha;
            color[offset + 1] = fragment.color[1] * alpha;
            color[offset + 2] = fragment.color[2] * alpha;
            color[offset + 3] = alpha;

            if (aov) {
              const o = s * 3;

              if (info) {
                modS[o] = info.modulation![0] * alpha; modS[o + 1] = info.modulation![1] * alpha;
                modS[o + 2] = info.modulation![2] * alpha;
                indS[o] = info.indirect![0]; indS[o + 1] = info.indirect![1]; indS[o + 2] = info.indirect![2];
                weightS[s] = alpha;
              } else {
                modS[o] = modS[o + 1] = modS[o + 2] = 0; weightS[s] = 0;
              }

              // The first sample picks the surface that describes the pixel for the denoiser.
              if (s === 0) {
                aov.depth[pixel] = info ? this.viewDepth(surface.position) : 0;
                aov.normal.set(fragmentNormal, pixel * 3);
                aov.variance[pixel] = info ? variance : 0;
              }
            }
          }
        }
      }

      for (let s = 0; s < S; s++) {
        const slot = s;

        if (entries) {
          const visible = entries.filter(e => e.mask & (1 << s))
            .map(e => ({ e, z: e.depth >= 0 ? e.depth : e.depths![s] }));

          visible.sort((a, b) => a.e.order - b.e.order || b.z - a.z);

          for (const { e, z } of visible) {
            const m = e.material, f = e.fragment;
            if (m.flag("depthTest", true) && !depthPass(m.number("depthFunc", 3), z, depth[slot]))
              continue;
            if (m.flag("depthWrite", true))
              depth[slot] = z;
            if (!m.flag("colorWrite", true))
              continue;
            const offset = slot * 4, dst = Array.from(color.subarray(offset, offset + 3)), da = color[offset + 3];
            const a = f.alpha, blending = m.number("blending", 1);

            // Only diffuse gives way to the attenuated background; specular and emissive stay on top, as in Three.
            // The refracted lookup replaces what lies straight behind the pixel whenever the scene can be traced.
            const behind = mulColor(f.transmitted ?? dst, f.attenuation);
            let rgb = f.transmission ? add(sub(f.color, f.diffuse), mix(f.diffuse, behind, f.transmission)) : f.color;

            if (aov) {
              // What lies beneath loses weight in the blend; the translucent's own indirect light stays in the rest.
              const o = slot * 3;
              if (blending === 0) {
                modS[o] = modS[o + 1] = modS[o + 2] = 0; weightS[slot] = 0;
              } else if (blending === 4)
                for (let c = 0; c < 3; c++) modS[o + c] *= 1 - a + a * rgb[c];
              else if (blending !== 2 && blending !== 3)
                for (let c = 0; c < 3; c++) modS[o + c] *= 1 - a;
            }

            // Built-in materials premultiply in their own shader, so only custom shaders arrive premultiplied.
            const source = f.premultiplied ? rgb : scale(rgb, a);

            if (blending === 2)
              rgb = add(dst, source);
            else if (blending === 3)
              rgb = sub(dst, scale(rgb, a)).map(v => Math.max(0, v));
            else if (blending === 4)
              rgb = mulColor(dst, mix([1, 1, 1], rgb, a));
            else if (blending !== 0)
              rgb = add(source, scale(dst, 1 - a));
            color.set([...rgb, blending === 0 ? 1 : a + da * (1 - a)], offset);
          }
        }

        if (this.volumetric.active) {
          const px = x + jitters[s][0], py = y + jitters[s][1];

          const vp = transform4(this.inverseProjection, [px / this.width * 2 - 1,
            1 - py / this.height * 2, (Number.isFinite(depth[slot]) ? depth[slot] : 1) * 2 - 1, 1]);

          const point = vp.slice(0, 3).map(v => v / vp[3]);
          // Undo the off-axis lens projection before reconstructing the world-space ray.
          point[0] -= lens[0] * (-point[2] / this.settings.dofFocusDistance - 1);
          point[1] -= lens[1] * (-point[2] / this.settings.dofFocusDistance - 1);

          const origin = this.orthographic ? this.viewRay(px, py).origin
            : transform4(this.cameraWorld, [lens[0], lens[1], 0]).slice(0, 3);

          const ray = sub(transform4(this.cameraWorld, point), origin);

          const glow = this.volumetric.sample(origin, normalize(ray), Math.hypot(...ray),
            (p, d, dist) => !this.settings.shadows || !this.bvh
              || !this.bvh.occluded(p, d, dist, -1, this.shadowAccept),
            (52.9829189 * (((x + .5) * .06711056 + (this.height - y - .5) * .00583715) % 1)) % 1,
            // Light scattered in the medium crosses the same fog as the surfaces on its way to the camera.
            this.fog ? p => 1 - this.fogAmount(p) : undefined);

          const offset = slot * 4;
          const strength = Math.max(...glow);

          if (strength > 0) {
            const alpha = Math.max(color[offset + 3], strength);
            for (let c = 0; c < 3; c++) color[offset + c] += glow[c];
            color[offset + 3] = alpha;
          }
        }

        if (targetDepth)
          targetDepth[pixel] = Math.min(targetDepth[pixel], depth[slot]);
        sum[pixel * 4] += color[slot * 4];
        sum[pixel * 4 + 1] += color[slot * 4 + 1];
        sum[pixel * 4 + 2] += color[slot * 4 + 2];
        sum[pixel * 4 + 3] += color[slot * 4 + 3];

        if (aov) {
          const o = slot * 3, w = weightS[slot];

          for (let c = 0; c < 3; c++) {
            aov.modulation[pixel * 3 + c] += modS[o + c];
            aov.indirect[pixel * 3 + c] += indS[o + c] * w;
          }

          aov.weight[pixel] += w;
        }
      }

      onPixel?.(pixel);

      // End of row: a natural point for the worker to handle messages and publish the finished band.
      if ((pixel + 1) % bucket.width === 0) {
        yield;
        lastYield = performance.now();
      }
    }
  }
}

const mulColor = (a: number[], b: number[]) => a.map((v, i) => v * b[i]);
