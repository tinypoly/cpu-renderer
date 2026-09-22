import type { FrameSettings } from "./frameSettings.js";
import type { SerializedLight } from "./sceneSerialization.js";
import type { RayHit } from "./bvh.js";
import type { CpuEnvironment } from "./environment.js";
import type { CpuMaterial, LightingContext, Surface } from "./material.js";
import { add, clamp, cross, dot, halton, length, mix, normalize, scale, sub } from "./math.js";

/** Triangles of the prepared geometry: positions (9 per triangle) and face normals (3) in flat arrays. */
export interface GiTriangles {
  count: number;
  positions: Float64Array;
  faceNormals: Float64Array;
  material: (index: number) => CpuMaterial;
}

interface GiScene {
  triangles: GiTriangles;
  settings: FrameSettings;
  lights: SerializedLight[];
  environment: CpuEnvironment;
  intersect: (p: number[], d: number[], distance: number, ignore: number) => RayHit | null;
  surface: (hit: RayHit, direction: number[]) => Surface;
  origin: (p: number[], n: number[]) => number[];
}

interface Emitter { index: number; area: number; weight: number; cumulative: number; normal: number[] }

interface SurfaceHit { hit: RayHit; surface: Surface | null }

/** Indirect-light estimate at a point, with what the cache and the denoiser need besides the color. */
export interface GiEstimate {
  color: number[];
  /** Variance of the sample mean (luminance), already in intensity scale. */
  variance: number;
  /** Harmonic mean of first-bounce distances; Infinity when no ray hit anything. */
  radius: number;
}

interface PathStats { inverseDistance: number; hits: number }

const BLACK = [0, 0, 0];
const luminance = (c: number[]) => c[0] * .2126 + c[1] * .7152 + c[2] * .0722;
const powerWeight = (a: number, b: number) => a * a / (a * a + b * b);
// Dimensions per bounce: cosine (0, 1), emitter (2, 3, 4), environment (5, 6) and Russian roulette (7).
const DIMENSIONS_PER_BOUNCE = 8;

// Separate dimensions per bounce; a pixel seed never depends on worker/bucket order.
const primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53,
  59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131,
  137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223,
  227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317];

function random(sample: number, dimension: number, seed: number) {
  const v = Math.sin(seed * 7183.17 + dimension * 39.3467) * 47453.5453;

  return (halton(sample + 1, primes[dimension % primes.length]) + v - Math.floor(v)) % 1;
}

/** Diffuse path tracing with cosine sampling and next-event sampling of emissive triangles.
 * MIS combines both estimators without counting emission twice. Cost is O(samples * bounces),
 * never samples^bounces. Rasterization still supplies the primary visible surface.
 */
export class DiffuseGi {
  private emitters: Emitter[] = [];
  private emitterByTriangle = new Map<number, Emitter>();
  private totalWeight = 0;
  private lights: SerializedLight[];
  private continuation = new Map<CpuMaterial, number>();

  constructor(private scene: GiScene) {
    // Visibility along indirect paths is geometric, independent of artistic direct-shadow flags.
    this.lights = scene.lights.map(light => ({ ...light, castShadow: true }));

    for (let index = 0; index < scene.triangles.count; index++) {
      const m = scene.triangles.material(index);
      // Patched/animated emission is still found by cosine rays. Constant native emission
      // provides a useful, inexpensive importance distribution, including textured emitters.
      if (m.data.shader) continue;
      const power = luminance(m.vector("emissive", BLACK)) * m.number("emissiveIntensity", 1);
      if (power <= 0) continue;
      const [a, b, c] = this.corners(index);
      const area = length(cross(sub(b, a), sub(c, a))) * .5;
      if (area <= 1e-12) continue;
      const weight = power * area;
      this.totalWeight += weight;
      const emitter = { index, area, weight, cumulative: this.totalWeight, normal: this.geometricNormal(index) };
      this.emitters.push(emitter);
      this.emitterByTriangle.set(index, emitter);
    }
  }

  private geometricNormal(index: number): number[] {
    const normals = this.scene.triangles.faceNormals, o = index * 3;

    return [normals[o], normals[o + 1], normals[o + 2]];
  }

  private corners(index: number): number[][] {
    const p = this.scene.triangles.positions, o = index * 9;

    return [[p[o], p[o + 1], p[o + 2]], [p[o + 3], p[o + 4], p[o + 5]], [p[o + 6], p[o + 7], p[o + 8]]];
  }

  evaluate(source: number, position: number[], normal: number[], seed: number): number[] {
    return this.estimate(source, position, normal, seed).color;
  }

  estimate(source: number, position: number[], normal: number[], seed: number): GiEstimate {
    const { giSamples = 64, giBounces = 3, giIntensity = 1, giClamp = 0 } = this.scene.settings;
    if (giIntensity === 0) return { color: BLACK, variance: 0, radius: Infinity };
    let sum = [0, 0, 0], sumL = 0, sumL2 = 0;
    const stats: PathStats = { inverseDistance: 0, hits: 0 };

    for (let sample = 0; sample < giSamples; sample++) {
      let value = this.irradiance(source, position, normal, giBounces, 0, sample, seed, stats);
      const energy = luminance(value);
      // Clamp the complete path only, never every bounce (which would compound bias).
      if (giClamp > 0 && energy > giClamp) value = scale(value, giClamp / energy);
      sum = add(sum, value);
      const l = luminance(value);
      sumL += l; sumL2 += l * l;
    }

    const mean = sumL / giSamples;

    const variance = giSamples > 1
      ? Math.max(0, sumL2 / giSamples - mean * mean) / (giSamples - 1) * giIntensity * giIntensity : 0;

    return {
      color: scale(sum, giIntensity / giSamples), variance,
      radius: stats.hits ? stats.hits / stats.inverseDistance : Infinity,
    };
  }

  private context(source: number, seed: number, indirect: LightingContext["indirect"],
    transport: "diffuse" | "emission", emissionWeight = 1): LightingContext {
    return { lights: this.lights, settings: this.scene.settings, environment: this.scene.environment,
      occlusion: () => 1, indirect, transport, emissionWeight,
      visibility: (p, n, d, distance) => {
        const g = this.geometricNormal(source), geometric = dot(g, n) < 0 ? scale(g, -1) : g;
        let direction = d;
        const spread = Math.tan((this.scene.settings.shadowSoftness ?? 0) * Math.PI / 360);

        if (spread > 0) {
          const t = normalize(cross(Math.abs(d[1]) > .99 ? [1, 0, 0] : [0, 1, 0], d)), b = cross(d, t);
          const r = Math.sqrt(random(0, 0, seed)) * spread, angle = random(0, 1, seed) * 2 * Math.PI;
          direction = normalize(add(d, add(scale(t, r * Math.cos(angle)), scale(b, r * Math.sin(angle)))));
        }

        if (dot(direction, geometric) <= 0) return 0;

        return this.hit(this.scene.origin(p, geometric), direction, distance, source, seed) ? 0 : 1;
      } };
  }

  /** Alpha cutouts/discard do not consume a bounce. Blended coverage is sampled stochastically. */
  private hit(origin: number[], d: number[], distance: number, ignore: number, seed: number): SurfaceHit | null {
    let p = origin, remaining = distance;

    for (let layer = 0; layer < 64; layer++) {
      const hit = this.scene.intersect(p, d, remaining, ignore);
      if (!hit) return null;
      const material = this.scene.triangles.material(hit.index);
      // A material that always blocks skips building the surface just to check alpha.
      if (material.rayOpaque) return { hit, surface: null };
      const surface = this.scene.surface(hit, d);
      const emission = material.shade(surface, this.context(hit.index, seed, () => BLACK, "emission"));
      if (emission && emission.alpha > random(layer, 39, seed)) return { hit, surface };
      const next = this.scene.origin(surface.position, d);
      remaining -= length(sub(next, p));
      p = next; ignore = hit.index;
    }

    // Excessive transparent layering terminates conservatively instead of leaking the sky.
    throw new Error("CPU GI: more than 64 transparent layers along a ray.");
  }

  private lightPdf(emitter: Emitter, from: number[], to: number[], direction: number[]) {
    const cosine = Math.abs(dot(emitter.normal, direction));
    const delta = sub(to, from);

    return cosine > 1e-10 ? emitter.weight / this.totalWeight * dot(delta, delta) / (emitter.area * cosine) : 0;
  }

  private facesRay(index: number, surface: Surface) {
    const side = this.scene.triangles.material(index).number("side", 0);

    return side === 2 || (side === 1 ? !surface.frontFacing : surface.frontFacing);
  }

  /** Probability of the path continuing at a material: a dark surface returns little, so it stops earlier. */
  private continuationProbability(material: CpuMaterial) {
    let q = this.continuation.get(material);

    if (q === undefined) {
      q = clamp(luminance(material.vector("color", [1, 1, 1])), .1, .95);
      this.continuation.set(material, q);
    }

    return q;
  }

  private irradiance(source: number, position: number[], normal: number[], remaining: number,
    depth: number, sample: number, seed: number, stats?: PathStats): number[] {
    if (remaining === 0) return BLACK;
    const g = this.geometricNormal(source);
    const geometric = dot(g, normal) < 0 ? scale(g, -1) : g;
    const origin = this.scene.origin(position, geometric), dimension = depth * DIMENSIONS_PER_BOUNCE;
    let result = this.sampleEmitter(source, origin, normal, geometric, sample, dimension, seed);
    const environment = this.scene.environment, rotation = this.scene.settings.environmentRotation;
    const importance = Boolean(environment.importance);
    if (importance)
      result = add(result, this.sampleEnvironment(source, origin, normal, geometric, sample, dimension, seed));
    const t = normalize(cross(Math.abs(normal[1]) > .99 ? [1, 0, 0] : [0, 1, 0], normal));
    const b = cross(normal, t), u = random(sample, dimension, seed);
    const phi = random(sample, dimension + 1, seed) * 2 * Math.PI;
    const r = Math.sqrt(u), cosine = Math.sqrt(1 - u);
    const d = add(add(scale(t, r * Math.cos(phi)), scale(b, r * Math.sin(phi))), scale(normal, cosine));
    if (dot(d, geometric) <= 0) return result;
    const found = this.hit(origin, d, Infinity, source, seed + sample * .754877 + depth * .56984);

    if (!found) {
      let sky = scale(environment.sample(d, rotation), this.scene.settings.environmentIntensity);
      // With map sampling, both strategies share the sky through the power heuristic.
      if (importance) sky = scale(sky, powerWeight(cosine / Math.PI, environment.pdf(d, rotation)));

      // Ambient/hemisphere lights are distant fill: visibility must apply to them too.
      for (const light of this.scene.lights) {
        if (light.type === "AmbientLight") sky = add(sky, scale(light.color, light.intensity / Math.PI));
        if (light.type === "HemisphereLight") sky = add(sky, scale(mix(light.groundColor, light.color,
          dot(d, normalize(light.position)) * .5 + .5), light.intensity / Math.PI));
      }

      return add(result, sky);
    }

    if (stats) {
      stats.inverseDistance += 1 / Math.max(found.hit.distance, 1e-6);
      stats.hits++;
    }

    const { hit } = found, surface = found.surface ?? this.scene.surface(hit, d);
    // Back faces block paths but do not emit/reflect on the unpainted side.
    if (!this.facesRay(hit.index, surface)) return result;
    const emitter = this.emitterByTriangle.get(hit.index);

    const emissionWeight = emitter ? powerWeight(cosine / Math.PI,
      this.lightPdf(emitter, origin, surface.position, d)) : 1;

    const material = this.scene.triangles.material(hit.index);
    // Russian roulette from the second bounce: the path continues with an albedo-based probability and is reweighted.
    let weight = 1;

    if (depth >= 1 && remaining > 1) {
      const q = this.continuationProbability(material);
      weight = random(sample, dimension + 7, seed) < q ? 1 / q : 0;
    }

    const ctx = this.context(hit.index, seed + sample + depth, weight === 0 ? () => BLACK
      : (p, n) => scale(this.irradiance(hit.index, p, n, remaining - 1, depth + 1, sample, seed), weight),
    "diffuse", emissionWeight);

    const shaded = material.shade(surface, ctx);
    if (shaded) result = add(result, shaded.color);

    return result;
  }

  private sampleEmitter(source: number, origin: number[], n: number[], geometric: number[],
    sample: number, dimension: number, seed: number): number[] {
    if (!this.emitters.length) return BLACK;
    const pick = random(sample, dimension + 2, seed) * this.totalWeight;
    let low = 0, high = this.emitters.length - 1;

    while (low < high) {
      const mid = (low + high) >>> 1;
      if (pick < this.emitters[mid].cumulative) high = mid; else low = mid + 1;
    }

    const emitter = this.emitters[low];
    if (emitter.index === source) return BLACK;
    const r = Math.sqrt(random(sample, dimension + 3, seed)), v = random(sample, dimension + 4, seed);
    const [a, b, c] = this.corners(emitter.index);
    const point = add(add(scale(a, 1 - r), scale(b, r * (1 - v))), scale(c, r * v));
    const delta = sub(point, origin), distance = length(delta), d = normalize(delta), cosine = dot(n, d);
    if (cosine <= 0 || dot(geometric, d) <= 0 || distance < 1e-7) return BLACK;
    const surface = this.scene.surface({ index: emitter.index, u: r * (1 - v), v: r * v, distance }, d);
    if (!this.facesRay(emitter.index, surface)) return BLACK;
    const pdf = this.lightPdf(emitter, origin, point, d);
    if (pdf <= 0) return BLACK;
    // Stop just before the emitter, allowing cutouts/coverage on intervening surfaces.
    const endpointBias = length(sub(this.scene.origin(point, d), point));
    if (this.hit(origin, d, Math.max(0, distance - endpointBias), source, seed + sample * .3287)) return BLACK;

    const emission = this.scene.triangles.material(emitter.index).shade(surface,
      this.context(emitter.index, seed, () => BLACK, "emission"));

    return emission ? scale(emission.color, emission.alpha * cosine / (Math.PI * pdf)
      * powerWeight(pdf, cosine / Math.PI)) : BLACK;
  }

  /** Samples an environment direction by map luminance; the power heuristic shares it with cosine sampling. */
  private sampleEnvironment(source: number, origin: number[], n: number[], geometric: number[],
    sample: number, dimension: number, seed: number): number[] {
    const environment = this.scene.environment, rotation = this.scene.settings.environmentRotation;

    const picked = environment.sampleDirection(
      random(sample, dimension + 5, seed), random(sample, dimension + 6, seed), rotation);

    if (!picked || picked.pdf <= 0) return BLACK;
    const d = picked.direction, cosine = dot(n, d);
    if (cosine <= 0 || dot(geometric, d) <= 0) return BLACK;
    if (this.hit(origin, d, Infinity, source, seed + sample * .1931)) return BLACK;
    const radiance = scale(environment.sample(d, rotation), this.scene.settings.environmentIntensity);

    return scale(radiance, cosine / Math.PI / picked.pdf * powerWeight(picked.pdf, cosine / Math.PI));
  }
}
