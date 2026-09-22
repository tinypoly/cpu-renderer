import type { Bucket } from "./rasterizer.js";
import { sharedFloat32 } from "./sharedBuffers.js";

/**
 * Whole-image buffers the denoiser reads after the last bucket: accumulated color (premultiplied, with alpha),
 * indirect-light modulation (diffuse albedo times coverage), the indirect light itself, normal, view depth
 * and variance. Zero depth marks a pixel with no opaque surface receiving indirect light.
 */
export interface FrameAov {
  width: number;
  height: number;
  color: Float32Array;
  modulation: Float32Array;
  indirect: Float32Array;
  normal: Float32Array;
  depth: Float32Array;
  variance: Float32Array;
}

/** Per-pixel offsets when a bucket travels packed (outside shared memory). */
export const AOV_STRIDE = 15;
const ITERATIONS = 5;
const KERNEL = [1 / 16, 1 / 4, 3 / 8, 1 / 4, 1 / 16];
const SIGMA_DEPTH = 1, SIGMA_LUMINANCE = 4;

const luminance = (data: Float32Array, i: number) =>
  data[i * 3] * .2126 + data[i * 3 + 1] * .7152 + data[i * 3 + 2] * .0722;

export function allocateFrameAov(width: number, height: number): FrameAov {
  const count = width * height;

  return {
    width, height, color: sharedFloat32(count * 4), modulation: sharedFloat32(count * 3),
    indirect: sharedFloat32(count * 3), normal: sharedFloat32(count * 3), depth: sharedFloat32(count),
    variance: sharedFloat32(count),
  };
}

export function packBucketAov(aov: FrameAov, bucket: Bucket): Float32Array<ArrayBuffer> {
  const out = new Float32Array(bucket.width * bucket.height * AOV_STRIDE);
  let cursor = 0;

  for (let y = 0; y < bucket.height; y++)
    for (let x = 0; x < bucket.width; x++) {
      const i = (bucket.y + y) * aov.width + bucket.x + x;
      out.set(aov.color.subarray(i * 4, i * 4 + 4), cursor);
      out.set(aov.modulation.subarray(i * 3, i * 3 + 3), cursor + 4);
      out.set(aov.indirect.subarray(i * 3, i * 3 + 3), cursor + 7);
      out.set(aov.normal.subarray(i * 3, i * 3 + 3), cursor + 10);
      out[cursor + 13] = aov.depth[i];
      out[cursor + 14] = aov.variance[i];
      cursor += AOV_STRIDE;
    }

  return out;
}

export function unpackBucketAov(aov: FrameAov, bucket: Bucket, data: Float32Array) {
  let cursor = 0;

  for (let y = 0; y < bucket.height; y++)
    for (let x = 0; x < bucket.width; x++) {
      const i = (bucket.y + y) * aov.width + bucket.x + x;
      aov.color.set(data.subarray(cursor, cursor + 4), i * 4);
      aov.modulation.set(data.subarray(cursor + 4, cursor + 7), i * 3);
      aov.indirect.set(data.subarray(cursor + 7, cursor + 10), i * 3);
      aov.normal.set(data.subarray(cursor + 10, cursor + 13), i * 3);
      aov.depth[i] = data[cursor + 13];
      aov.variance[i] = data[cursor + 14];
      cursor += AOV_STRIDE;
    }
}

/** Cosine between normals raised to the 64th power, by repeated squaring. */
function normalWeight(c: number) {
  if (c <= 0) return 0;
  let v = c * c; v *= v; v *= v; v *= v; v *= v; v *= v;

  return v;
}

/**
 * À-trous filter guided by normal, depth and variance (in the spirit of SVGF): five passes with doubling step,
 * the variance is filtered alongside, and luminance only blends with neighbors within its own noise level.
 * Returns the filtered indirect light, three values per pixel; pixels without a surface pass through unchanged.
 *
 * `correlation` is the distance, in pixels, over which neighbors share their noise (the reach of an irradiance cache
 * record; zero when every pixel has its own estimate). Averaging pixels that share their error does not reduce it,
 * so the variance only shrinks once the filter step reaches that distance. Shrinking it earlier closes the
 * luminance threshold after the first passes and leaves the records' low-frequency noise as blotches.
 */
export function* denoiseIndirect(aov: FrameAov, correlation = 0): Generator<void, Float32Array> {
  const { width, height, depth, normal } = aov, count = width * height;
  let color = Float32Array.from(aov.indirect), variance = Float32Array.from(aov.variance);
  let outColor = new Float32Array(count * 3), outVariance = new Float32Array(count);
  // Per-pixel depth gradient: separates surfaces that deviate from a continuous slope.
  const gradient = new Float32Array(count);

  for (let y = 0; y < height; y++)
    for (let x = 0; x < width; x++) {
      const i = y * width + x, z = depth[i];
      if (!(z > 0)) continue;
      let g = 0;

      for (const j of [i - 1, i + 1, i - width, i + width]) {
        if (j < 0 || j >= count || !(depth[j] > 0)) continue;
        if ((j === i - 1 && x === 0) || (j === i + 1 && x === width - 1)) continue;
        g = Math.max(g, Math.abs(z - depth[j]));
      }

      gradient[i] = g;
    }

  let lastYield = performance.now();

  for (let iteration = 0; iteration < ITERATIONS; iteration++) {
    const step = 1 << iteration, independent = step >= correlation;

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const i = y * width + x, z = depth[i];

        if (!(z > 0)) {
          outColor[i * 3] = color[i * 3];
          outColor[i * 3 + 1] = color[i * 3 + 1];
          outColor[i * 3 + 2] = color[i * 3 + 2];
          outVariance[i] = variance[i];
          continue;
        }

        // Local variance smoothed over 3x3: the luminance threshold must not depend on a single pixel.
        let local = 0, localWeight = 0;

        for (let dy = -1; dy <= 1; dy++) {
          const yy = y + dy;
          if (yy < 0 || yy >= height) continue;

          for (let dx = -1; dx <= 1; dx++) {
            const xx = x + dx;
            if (xx < 0 || xx >= width) continue;
            const j = yy * width + xx;
            if (!(depth[j] > 0)) continue;
            const w = (dx === 0 ? .5 : .25) * (dy === 0 ? .5 : .25);
            local += variance[j] * w; localWeight += w;
          }
        }

        const sigma = SIGMA_LUMINANCE * Math.sqrt(localWeight > 0 ? local / localWeight : variance[i]) + 1e-4;
        const l = luminance(color, i), nx = normal[i * 3], ny = normal[i * 3 + 1], nz = normal[i * 3 + 2];
        const slope = gradient[i];
        let sumW = 0, r = 0, g = 0, b = 0, v = 0;

        for (let ky = -2; ky <= 2; ky++) {
          const yy = y + ky * step;
          if (yy < 0 || yy >= height) continue;

          for (let kx = -2; kx <= 2; kx++) {
            const xx = x + kx * step;
            if (xx < 0 || xx >= width) continue;
            const j = yy * width + xx, zj = depth[j];
            if (!(zj > 0)) continue;
            const h = KERNEL[ky + 2] * KERNEL[kx + 2];
            const wn = normalWeight(nx * normal[j * 3] + ny * normal[j * 3 + 1] + nz * normal[j * 3 + 2]);

            const wz = Math.exp(-Math.abs(z - zj)
              / (SIGMA_DEPTH * slope * (Math.abs(kx) + Math.abs(ky)) * step + 1e-3 * z + 1e-6));

            const wl = Math.exp(-Math.abs(l - luminance(color, j)) / sigma);
            const w = h * wn * wz * wl;
            sumW += w;
            r += color[j * 3] * w; g += color[j * 3 + 1] * w; b += color[j * 3 + 2] * w;
            v += variance[j] * (independent ? w * w : w);
          }
        }

        outColor[i * 3] = r / sumW; outColor[i * 3 + 1] = g / sumW; outColor[i * 3 + 2] = b / sumW;
        outVariance[i] = independent ? v / (sumW * sumW) : v / sumW;
      }

      if (performance.now() - lastYield > 8) {
        yield;
        lastYield = performance.now();
      }
    }

    [color, outColor] = [outColor, color];
    [variance, outVariance] = [outVariance, variance];
  }

  return color;
}
