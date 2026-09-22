import { Color, FloatType } from "three";
import { HDRLoader } from "three/examples/jsm/loaders/HDRLoader.js";
import type { RenderEnvironment } from "./settings.js";
import type { SerializedTexture } from "./sceneSerialization.js";
import type { ProceduralSky } from "./extensions.js";
import { CpuShader, type Value } from "./glsl.js";
import { add, clamp, cross, halton, mix, normalize, scale, sub, type Vec3 } from "./math.js";
import { type TextureSampling, sampleTexture } from "./texture.js";
import { sharedFloat32, sharedFloat64 } from "./sharedBuffers.js";

/** Equirectangular map stored as a grid of linear radiances, three values per texel. */
export interface RadianceGrid {
  width: number;
  height: number;
  texels: Float64Array;
}

/**
 * Distribution for sampling directions proportionally to the map's luminance: CDFs of the rows and, per row, of the
 * columns, over the base grid. Only exists when the map has peaks (sun, lamps) that cosine sampling rarely finds.
 */
export interface EnvironmentImportance {
  width: number;
  height: number;
  /** Normalized CDF over rows (weight is luminance times the sine of the colatitude). */
  rows: Float64Array;
  /** Normalized CDF of the columns within each row, `height * width` entries. */
  columns: Float64Array;
  /** Luminance of each texel, used for the density of a direction. */
  luminance: Float64Array;
  /** Sum of luminance times sine over the grid; normalizes the density. */
  total: number;
}

/** Already convolved state, without compiled shaders: moves between workers without copying the shared buffers. */
export interface CpuEnvironmentSnapshot {
  texture: SerializedTexture | null;
  levels: RadianceGrid[];
  diffuse: RadianceGrid | null;
  importance: EnvironmentImportance | null;
  top: number[];
  bottom: number[];
  exponent: number;
  procedural?: ProceduralSky;
}

/** Roughness of the prefiltered levels (uniform spacing); below the first one it blends with the sharp map. */
const LEVEL_STEP = .25;
const LEVEL_SIZES = [[96, 48], [48, 24], [32, 16], [32, 16]];
const LEVEL_SAMPLES = [32, 48, 64, 64];
const BASE_WIDTH = 256, BASE_HEIGHT = 128;
/** Ratio between the brightest texel and the mean above which importance sampling the map pays off. */
const IMPORTANCE_PEAK_RATIO = 8;
const texelLuminance = (c: Vec3) => c[0] * .2126 + c[1] * .7152 + c[2] * .0722;

const direction = (u: number, v: number): Vec3 => {
  const phi = (u - .5) * 2 * Math.PI, theta = v * Math.PI;

  return [Math.sin(theta) * Math.cos(phi), Math.cos(theta), Math.sin(theta) * Math.sin(phi)];
};

function lookup(grid: RadianceGrid, d: Vec3): Vec3 {
  const u = (Math.atan2(d[2], d[0]) / (2 * Math.PI) + .5) * grid.width - .5;
  const v = Math.acos(clamp(d[1], -1, 1)) / Math.PI * grid.height - .5;
  const x = Math.floor(u), y = Math.floor(v), texels = grid.texels;

  const texel = (a: number, b: number): Vec3 => {
    const i = (clamp(b, 0, grid.height - 1) * grid.width + ((a % grid.width) + grid.width) % grid.width) * 3;

    return [texels[i], texels[i + 1], texels[i + 2]];
  };

  return mix(mix(texel(x, y), texel(x + 1, y), u - x), mix(texel(x, y + 1), texel(x + 1, y + 1), u - x), v - y);
}

/** The base grid and its 2 × 2 box-filtered reductions, down to a few texels. */
function mipChain(base: RadianceGrid): RadianceGrid[] {
  const chain = [base];

  while (chain[chain.length - 1].width > 8) {
    const { width, height, texels } = chain[chain.length - 1], w = width / 2, h = height / 2;
    const next = new Float64Array(w * h * 3);

    for (let y = 0; y < h; y++)
      for (let x = 0; x < w; x++)
        for (let c = 0; c < 3; c++) {
          const at = (dx: number, dy: number) => texels[((2 * y + dy) * width + 2 * x + dx) * 3 + c];
          next[(y * w + x) * 3 + c] = (at(0, 0) + at(1, 0) + at(0, 1) + at(1, 1)) / 4;
        }

    chain.push({ width: w, height: h, texels: next });
  }

  return chain;
}

/**
 * Filtered importance sampling (Křivánek and Colbert): a sample standing for `solidAngle` steradians reads the mip
 * level whose texels cover about that much. Small, very bright emitters, as in a studio map, then average out
 * instead of leaving blotches where a few samples happened to hit them.
 */
function lookupFiltered(chain: RadianceGrid[], d: Vec3, solidAngle: number): Vec3 {
  const texel = 4 * Math.PI / (chain[0].width * chain[0].height);
  const lod = clamp(.5 * Math.log2(solidAngle / texel) + 1, 0, chain.length - 1), low = Math.floor(lod);
  if (low === lod || low === chain.length - 1)
    return lookup(chain[low], d);

  return mix(lookup(chain[low], d), lookup(chain[low + 1], d), lod - low);
}

function tangentFrame(n: Vec3) {
  const t = normalize(cross(Math.abs(n[1]) > .99 ? [1, 0, 0] : [0, 1, 0], n));

  return { t, b: cross(n, t) };
}

/** Convolves the base grid with a GGX lobe centered on the normal (split sum, like Three's PMREM). */
function prefilter(
  chain: RadianceGrid[], roughness: number, width: number, height: number, samples: number): RadianceGrid {
  const alpha = roughness * roughness, texels = sharedFloat64(width * height * 3);

  for (let y = 0; y < height; y++)
    for (let x = 0; x < width; x++) {
      const n = direction((x + .5) / width, (y + .5) / height), { t, b } = tangentFrame(n);
      let sum: Vec3 = [0, 0, 0], weight = 0;

      for (let i = 1; i <= samples; i++) {
        const phi = halton(i, 2) * 2 * Math.PI, u = halton(i, 3);
        const cosTheta = Math.sqrt((1 - u) / (1 + (alpha * alpha - 1) * u));
        const sinTheta = Math.sqrt(1 - cosTheta * cosTheta);
        const h = add(add(scale(t, sinTheta * Math.cos(phi)), scale(b, sinTheta * Math.sin(phi))), scale(n, cosTheta));
        const l = sub(scale(h, 2 * cosTheta), n), nl = l[0] * n[0] + l[1] * n[1] + l[2] * n[2];
        if (nl <= 0)
          continue;
        // With the view along the normal, the reflected direction's density is D / 4.
        const lobe = cosTheta * cosTheta * (alpha * alpha - 1) + 1, pdf = alpha * alpha / (Math.PI * lobe * lobe) / 4;
        sum = add(sum, scale(lookupFiltered(chain, l, 1 / (samples * pdf)), nl));
        weight += nl;
      }

      texels.set(weight > 0 ? scale(sum, 1 / weight) : [0, 0, 0], (y * width + x) * 3);
    }

  return { width, height, texels };
}

/** Luminance CDFs of the base grid; null when the map is too smooth to be worth the extra ray. */
function buildImportance(base: RadianceGrid): EnvironmentImportance | null {
  const { width, height, texels } = base, count = width * height;
  const luminance = sharedFloat64(count), columns = sharedFloat64(count), rows = sharedFloat64(height);
  let total = 0, peak = 0, sum = 0;

  for (let y = 0; y < height; y++) {
    const sinTheta = Math.sin((y + .5) / height * Math.PI);
    let rowSum = 0;

    for (let x = 0; x < width; x++) {
      const i = y * width + x, l = texelLuminance([texels[i * 3], texels[i * 3 + 1], texels[i * 3 + 2]]);
      luminance[i] = l;
      rowSum += l * sinTheta;
      columns[i] = rowSum;
      peak = Math.max(peak, l); sum += l;
    }

    for (let x = 0; x < width; x++)
      columns[y * width + x] = rowSum > 0 ? columns[y * width + x] / rowSum : (x + 1) / width;
    total += rowSum;
    rows[y] = total;
  }

  if (!(total > 0) || peak < IMPORTANCE_PEAK_RATIO * (sum / count))
    return null;
  for (let y = 0; y < height; y++) rows[y] /= total;

  return { width, height, rows, columns, luminance, total };
}

/** First CDF index whose value exceeds `u`. */
function searchCdf(cdf: Float64Array, start: number, count: number, u: number) {
  let low = start, high = start + count - 1;

  while (low < high) {
    const mid = (low + high) >>> 1;
    if (u < cdf[mid]) high = mid; else low = mid + 1;
  }

  return low;
}

/** Irradiance (cosine lobe) from the base grid. */
function irradiance(chain: RadianceGrid[], width: number, height: number, samples: number): RadianceGrid {
  const texels = sharedFloat64(width * height * 3);

  for (let y = 0; y < height; y++)
    for (let x = 0; x < width; x++) {
      const n = direction((x + .5) / width, (y + .5) / height), { t, b } = tangentFrame(n);
      let color: Vec3 = [0, 0, 0];

      for (let i = 1; i <= samples; i++) {
        const r = Math.sqrt(halton(i, 2)), a = halton(i, 3) * 2 * Math.PI;
        const cosine = Math.sqrt(1 - r * r);
        const d = add(add(scale(t, r * Math.cos(a)), scale(b, r * Math.sin(a))), scale(n, cosine));
        // Cosine-weighted directions have density cos / pi.
        color = add(color, scale(lookupFiltered(chain, d, Math.PI / (samples * Math.max(cosine, .05))), 1 / samples));
      }

      texels.set(color, (y * width + x) * 3);
    }

  return { width, height, texels };
}

/** A GPU renderer samples the sky as a cube; the CPU samples the same directions in its linear panorama. */
function foregroundShader(source: string | undefined): CpuShader | undefined {
  if (!source)
    return undefined;
  const replaced = source.replace(/\btextureCube\s*\(\s*uSkyMap\s*,/g, "cpuSkyMap(");

  return new CpuShader(`
        varying vec3 vDirection;
        uniform sampler2D uCpuSky;
        vec4 cpuSkyMap(vec3 direction) {
          vec3 d = normalize(direction);
          return texture2D(uCpuSky, vec2(atan(d.z, d.x) / 6.283185307179586 + 0.5,
            acos(clamp(d.y, -1.0, 1.0)) / 3.141592653589793));
        }
        ${replaced}
        void main() { gl_FragColor = vec4(skyForeground(normalize(vDirection)), 1.0); }
      `);
}

export class CpuEnvironment {
  private texture: SerializedTexture | null = null;
  private levels: RadianceGrid[] = [];
  private diffuse: RadianceGrid | null = null;
  /** Sampling distribution of the map; null disables environment importance sampling in global illumination. */
  importance: EnvironmentImportance | null = null;
  private top: number[] = [1, 1, 1];
  private bottom: number[] = [0, 0, 0];
  private exponent = 2;
  private procedural?: ProceduralSky;
  private foreground?: CpuShader;

  /** Prepare the smooth atmosphere once; celestial bodies stay sharp at output resolution. */
  static *fromProcedural(descriptor: ProceduralSky, width = 128): Generator<void, CpuEnvironment> {
    const env = new CpuEnvironment();
    env.procedural = descriptor;
    const height = width / 2;
    if (!Number.isInteger(width) || width < 8 || width > 512 || !Number.isInteger(height))
      throw new Error("CPU renderer: invalid procedural sky resolution.");
    const shader = new CpuShader(descriptor.fragmentShader);
    const data = sharedFloat32(width * height * 4);

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const result = shader.run({ ...descriptor.uniforms, vDirection: direction((x + .5) / width, (y + .5) / height),
          gl_FragCoord: [x + .5, y + .5, 0, 1], gl_FragColor: [0, 0, 0, 1] },
        { texture: () => {
          throw new Error("CPU renderer: unbound procedural sky texture.");
        } });

        const color = result?.gl_FragColor;
        if (!Array.isArray(color) || color.length !== 4 || !color.every(v => typeof v === "number" && Number.isFinite(v)))
          throw new Error("CPU renderer: procedural sky must output finite RGBA radiance.");
        data.set(color as number[], (y * width + x) * 4);
      }

      yield;
    }

    env.texture = { id: "procedural-sky", width, height, data, colorSpace: "srgb-linear",
      wrapS: 1000, wrapT: 1001, matrix: [1, 0, 0, 0, 1, 0, 0, 0, 1], flipY: false, channel: 0, nearest: false };
    env.foreground = foregroundShader(descriptor.foregroundShader);
    env.convolve();

    return env;
  }

  snapshot(): CpuEnvironmentSnapshot {
    return {
      texture: this.texture, levels: this.levels, diffuse: this.diffuse, importance: this.importance,
      top: this.top, bottom: this.bottom, exponent: this.exponent, procedural: this.procedural,
    };
  }

  /** Environment convolved by another worker: only the sky shader is compiled again. */
  static fromSnapshot(snapshot: CpuEnvironmentSnapshot): CpuEnvironment {
    const env = new CpuEnvironment();
    env.texture = snapshot.texture;
    env.levels = snapshot.levels;
    env.diffuse = snapshot.diffuse;
    env.importance = snapshot.importance;
    env.top = snapshot.top;
    env.bottom = snapshot.bottom;
    env.exponent = snapshot.exponent;
    env.procedural = snapshot.procedural;
    env.foreground = foregroundShader(snapshot.procedural?.foregroundShader);

    return env;
  }

  get backgroundVisible(): boolean {
    return this.procedural?.background ?? true;
  }

  /** Background brightness is independent of the sky's illumination strength. */
  sampleBackground(d: number[], rotation: number): Vec3 {
    if (!this.procedural) return this.sample(d, rotation);
    if (!this.foreground) return this.raw(d);

    const result = this.foreground.run({ ...this.procedural.uniforms, vDirection: d,
      uCpuSky: { texture: this.texture!.id }, gl_FragColor: [0, 0, 0, 1] },
    { texture: (_sampler: Value, uv: number[], sampling?: TextureSampling) =>
      sampleTexture(this.texture!, uv, false, sampling) });

    const color = result?.gl_FragColor;
    if (!Array.isArray(color) || color.length !== 4 || !color.every(v => typeof v === "number" && Number.isFinite(v)))
      throw new Error("CPU renderer: sky foreground must output finite RGBA radiance.");

    return color.slice(0, 3) as Vec3;
  }

  /** Environment from a linear equirectangular map already in memory (tests and fixtures). */
  static fromTexture(texture: SerializedTexture): CpuEnvironment {
    const env = new CpuEnvironment();
    env.texture = texture;
    env.convolve();

    return env;
  }

  static async load(descriptor: RenderEnvironment, signal?: AbortSignal) {
    const env = new CpuEnvironment();

    if (descriptor.kind === "hdri") {
      if (!descriptor.url)
        throw new Error("HDRI URL is missing.");
      const response = await fetch(descriptor.url, { signal });
      if (!response.ok)
        throw new Error(`HDRI: HTTP ${response.status}`);
      const parsed = new HDRLoader().setDataType(FloatType).parse(await response.arrayBuffer());
      if (!parsed.width || !parsed.height || !parsed.data)
        throw new Error("Invalid HDRI image.");
      const data = sharedFloat32((parsed.data as Float32Array).length);
      data.set(parsed.data as Float32Array);
      env.texture = {
        id: "environment",
        data,
        width: parsed.width,
        height: parsed.height,
        colorSpace: "srgb-linear",
        wrapS: 1000,
        wrapT: 1001,
        matrix: [1, 0, 0, 0, 1, 0, 0, 0, 1],
        flipY: false,
        channel: 0,
        nearest: false,
      };
    } else {
      env.top = new Color(descriptor.topColor ?? "#ffffff").toArray();
      env.bottom = new Color(descriptor.bottomColor ?? "#000000").toArray();
      env.exponent = descriptor.exponent ?? 2;
    }

    env.convolve();

    return env;
  }

  private raw(direction: number[]): Vec3 {
    if (!this.texture)
      return mix(this.bottom, this.top, ((direction[1] + 1) / 2) ** this.exponent);

    return sampleTexture(this.texture, [
      Math.atan2(direction[2], direction[0]) / (2 * Math.PI) + .5,
      Math.acos(clamp(direction[1], -1, 1)) / Math.PI,
    ], false).slice(0, 3) as Vec3;
  }

  /** Base grid downsampled by box averaging: acts as the mip level for the convolutions. */
  private base(): RadianceGrid {
    const texels = new Float64Array(BASE_WIDTH * BASE_HEIGHT * 3);

    for (let y = 0; y < BASE_HEIGHT; y++)
      for (let x = 0; x < BASE_WIDTH; x++) {
        if (!this.texture) {
          texels.set(this.raw(direction((x + .5) / BASE_WIDTH, (y + .5) / BASE_HEIGHT)), (y * BASE_WIDTH + x) * 3);
          continue;
        }

        const { width, height, data } = this.texture;
        const x0 = Math.floor(x * width / BASE_WIDTH), y0 = Math.floor(y * height / BASE_HEIGHT);
        const x1 = Math.max(x0 + 1, Math.floor((x + 1) * width / BASE_WIDTH));
        const y1 = Math.max(y0 + 1, Math.floor((y + 1) * height / BASE_HEIGHT));
        const sum: Vec3 = [0, 0, 0];

        for (let sy = y0; sy < y1; sy++)
          for (let sx = x0; sx < x1; sx++) {
            const offset = (Math.min(sy, height - 1) * width + Math.min(sx, width - 1)) * 4;
            sum[0] += data[offset]; sum[1] += data[offset + 1]; sum[2] += data[offset + 2];
          }

        texels.set(scale(sum, 1 / ((x1 - x0) * (y1 - y0))), (y * BASE_WIDTH + x) * 3);
      }

    return { width: BASE_WIDTH, height: BASE_HEIGHT, texels };
  }

  private convolve() {
    const base = this.base(), chain = mipChain(base);
    this.levels = LEVEL_SIZES.map(([width, height], index) =>
      prefilter(chain, LEVEL_STEP * (index + 1), width, height, LEVEL_SAMPLES[index]));
    this.diffuse = irradiance(chain, 32, 16, 64);
    this.importance = this.texture ? buildImportance(base) : null;
  }

  /** Effective rotation: a procedural sky ignores the HDR rotation so its sun stays aligned with the light. */
  private angle(rotation: number) {
    return this.procedural ? 0 : rotation * Math.PI / 180;
  }

  /** World direction sampled by the map's luminance, and its density per solid angle. */
  sampleDirection(u: number, v: number, rotation: number): { direction: Vec3; pdf: number } | null {
    const imp = this.importance;
    if (!imp) return null;
    const y = searchCdf(imp.rows, 0, imp.height, u);
    const rowStart = y === 0 ? 0 : imp.rows[y - 1], rowSpan = imp.rows[y] - rowStart;
    const x = searchCdf(imp.columns, y * imp.width, imp.width, v) - y * imp.width;
    const columnStart = x === 0 ? 0 : imp.columns[y * imp.width + x - 1];
    const columnSpan = imp.columns[y * imp.width + x] - columnStart;
    // The remainder of each random number jitters the direction within the chosen texel.
    const jy = rowSpan > 0 ? clamp((u - rowStart) / rowSpan, 0, .999999) : .5;
    const jx = columnSpan > 0 ? clamp((v - columnStart) / columnSpan, 0, .999999) : .5;
    const t = direction((x + jx) / imp.width, (y + jy) / imp.height);
    const angle = this.angle(rotation), c = Math.cos(angle), s = Math.sin(angle);

    return {
      direction: [t[0] * c + t[2] * s, t[1], -t[0] * s + t[2] * c],
      pdf: imp.luminance[y * imp.width + x] * imp.width * imp.height / (2 * Math.PI * Math.PI * imp.total),
    };
  }

  /** Density per solid angle with which `sampleDirection` produces a world direction. */
  pdf(d: number[], rotation: number): number {
    const imp = this.importance;
    if (!imp) return 0;
    const angle = this.angle(rotation), c = Math.cos(angle), s = Math.sin(angle);
    const t = normalize([d[0] * c - d[2] * s, d[1], d[0] * s + d[2] * c]);

    const x = ((Math.floor((Math.atan2(t[2], t[0]) / (2 * Math.PI) + .5) * imp.width) % imp.width) + imp.width)
      % imp.width;

    const y = clamp(Math.floor(Math.acos(clamp(t[1], -1, 1)) / Math.PI * imp.height), 0, imp.height - 1);

    return imp.luminance[y * imp.width + x] * imp.width * imp.height / (2 * Math.PI * Math.PI * imp.total);
  }

  sample(direction: number[], rotation: number, roughness = 0, diffuse = false): Vec3 {
    // Procedural sky directions also drive a world-space sun; HDR rotation must not detach it from its light.
    if (this.procedural) rotation = 0;
    const intensity = this.procedural?.intensity ?? 1;
    const angle = rotation * Math.PI / 180, c = Math.cos(angle), s = Math.sin(angle);
    const d = normalize([direction[0] * c - direction[2] * s, direction[1], direction[0] * s + direction[2] * c]);
    if (diffuse)
      return scale(this.diffuse ? lookup(this.diffuse, d) : this.raw(d), intensity);
    if (roughness <= 0 || !this.levels.length)
      return scale(this.raw(d), intensity);
    // Rough reflection: interpolates the two neighboring prefiltered levels (the sharp map counts as level zero).
    const position = clamp(roughness) / LEVEL_STEP, index = Math.floor(position);
    const lower = index === 0 ? this.raw(d) : lookup(this.levels[index - 1], d);
    const upper = lookup(this.levels[Math.min(index, this.levels.length - 1)], d);

    return scale(mix(lower, upper, position - index), intensity);
  }
}
