import { describe, expect, it } from "vitest";
import { allocateFrameAov, denoiseIndirect, packBucketAov, unpackBucketAov, type FrameAov } from "./denoise.js";

function drain<T>(job: Generator<void, T>): T {
  let step = job.next();
  while (!step.done) step = job.next();

  return step.value;
}

/** Flat image with deterministic noise; the right half is another surface, with a different normal and light. */
function noisyFrame(width = 32, height = 32): FrameAov {
  const aov = allocateFrameAov(width, height);
  let seed = 3;

  const noise = () => {
    seed = (seed * 1664525 + 1013904223) % 4294967296;

    return seed / 4294967296 - .5;
  };

  for (let y = 0; y < height; y++)
    for (let x = 0; x < width; x++) {
      const i = y * width + x, right = x >= width / 2, base = right ? 2 : .5;
      aov.indirect.set([base + noise() * .6, base + noise() * .6, base + noise() * .6], i * 3);
      aov.normal.set(right ? [1, 0, 0] : [0, 1, 0], i * 3);
      aov.depth[i] = 5 + y * .01;
      aov.variance[i] = .03;
      aov.modulation.set([.8, .8, .8], i * 3);
      aov.color.set([1, 1, 1, 1], i * 4);
    }

  // A corner with no surface: background, which the filter must neither touch nor use.
  aov.depth[0] = 0;
  aov.indirect.set([9, 9, 9], 0);

  return aov;
}

const stats = (data: Float32Array, indices: number[]) => {
  const values = indices.map(i => data[i * 3]);
  const mean = values.reduce((a, b) => a + b, 0) / values.length;

  return { mean, deviation: Math.sqrt(values.reduce((a, v) => a + (v - mean) ** 2, 0) / values.length) };
};

describe("indirect-light denoiser", () => {
  it("reduces variance on each surface without mixing them or touching the background", () => {
    const aov = noisyFrame(), width = aov.width;
    const left: number[] = [], right: number[] = [];
    for (let y = 4; y < 28; y++)
      for (let x = 4; x < 28; x++) (x < width / 2 - 2 ? left : x >= width / 2 + 2 ? right : []).push(y * width + x);
    const before = { left: stats(aov.indirect, left), right: stats(aov.indirect, right) };
    const filtered = drain(denoiseIndirect(aov));
    const after = { left: stats(filtered, left), right: stats(filtered, right) };
    expect(after.left.deviation).toBeLessThan(before.left.deviation / 3);
    expect(after.right.deviation).toBeLessThan(before.right.deviation / 3);
    expect(after.left.mean).toBeCloseTo(before.left.mean, 1);
    expect(after.right.mean).toBeCloseTo(before.right.mean, 1);
    // The edge between surfaces stays in place: the pixel next to the boundary keeps its own side's value.
    expect(filtered[(16 * width + width / 2 - 1) * 3]).toBeLessThan(1);
    expect(filtered[(16 * width + width / 2) * 3]).toBeGreaterThan(1.5);
    expect(filtered[0]).toBe(9);
  });
  it("smooths noise shared by blocks of pixels once it knows how far the blocks reach", () => {
    // Like irradiance cache records: every 8 x 8 block shares one noisy value and reports that value's variance.
    const width = 64, aov = allocateFrameAov(width, width), blocks = new Map<number, number>();
    let seed = 7;

    for (let y = 0; y < width; y++)
      for (let x = 0; x < width; x++) {
        const i = y * width + x, block = (y >> 3) * 8 + (x >> 3);

        if (!blocks.has(block)) {
          seed = (seed * 1664525 + 1013904223) % 4294967296;
          blocks.set(block, 1 + (seed / 4294967296 - .5) * .2);
        }

        aov.indirect.fill(blocks.get(block)!, i * 3, i * 3 + 3);
        aov.normal.set([0, 1, 0], i * 3);
        aov.depth[i] = 5;
        aov.variance[i] = .2 * .2 / 12;
      }

    const inner: number[] = [];
    for (let y = 16; y < 48; y++)
      for (let x = 16; x < 48; x++) inner.push(y * width + x);
    const before = stats(aov.indirect, inner).deviation;
    const unaware = stats(drain(denoiseIndirect(aov)), inner).deviation;
    const aware = stats(drain(denoiseIndirect(aov, 8)), inner).deviation;
    expect(aware).toBeLessThan(before / 3);
    expect(aware).toBeLessThan(unaware * .7);
  });
  it("packs and unpacks a bucket region losslessly", () => {
    const aov = noisyFrame(16, 8), copy = allocateFrameAov(16, 8);
    const bucket = { x: 4, y: 2, width: 8, height: 4, index: 0 };
    unpackBucketAov(copy, bucket, packBucketAov(aov, bucket));

    for (let y = 0; y < 8; y++)
      for (let x = 0; x < 16; x++) {
        const i = y * 16 + x, inside = x >= 4 && x < 12 && y >= 2 && y < 6;
        expect(copy.depth[i]).toBe(inside ? aov.depth[i] : 0);
        expect(copy.indirect[i * 3]).toBe(inside ? aov.indirect[i * 3] : 0);
        expect(copy.color[i * 4 + 3]).toBe(inside ? 1 : 0);
      }
  });
});
