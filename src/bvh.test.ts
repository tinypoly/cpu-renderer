import { describe, expect, it } from "vitest";
import { ShadowBvh, type ShadowTriangle } from "./bvh.js";

/** Deterministic generator for the test triangles. */
function rng(seed: number) {
  return () => {
    seed = (seed * 1664525 + 1013904223) % 4294967296;

    return seed / 4294967296;
  };
}

function randomTriangles(count: number, seed: number): ShadowTriangle[] {
  const random = rng(seed), triangles: ShadowTriangle[] = [];

  for (let i = 0; i < count; i++) {
    const cx = random() * 20 - 10, cy = random() * 20 - 10, cz = random() * 20 - 10;
    const corner = () => [cx + random() - .5, cy + random() - .5, cz + random() - .5];
    triangles.push({ positions: [corner(), corner(), corner()], castShadow: random() > .3 });
  }

  return triangles;
}

/** Brute-force closest hit (Möller-Trumbore), to check the tree against. */
function bruteForce(triangles: ShadowTriangle[], origin: number[], d: number[], shadowOnly: boolean) {
  let best: { index: number; distance: number } | null = null;
  triangles.forEach((triangle, index) => {
    if (shadowOnly && !triangle.castShadow) return;
    const [a, b, c] = triangle.positions;
    const e1 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]], e2 = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    const h = [d[1] * e2[2] - d[2] * e2[1], d[2] * e2[0] - d[0] * e2[2], d[0] * e2[1] - d[1] * e2[0]];
    const det = e1[0] * h[0] + e1[1] * h[1] + e1[2] * h[2];
    if (Math.abs(det) < 1e-10) return;
    const s = [origin[0] - a[0], origin[1] - a[1], origin[2] - a[2]];
    const u = (s[0] * h[0] + s[1] * h[1] + s[2] * h[2]) / det;
    if (u < 0 || u > 1) return;
    const q = [s[1] * e1[2] - s[2] * e1[1], s[2] * e1[0] - s[0] * e1[2], s[0] * e1[1] - s[1] * e1[0]];
    const v = (d[0] * q[0] + d[1] * q[1] + d[2] * q[2]) / det;
    if (v < 0 || u + v > 1) return;
    const t = (e2[0] * q[0] + e2[1] * q[1] + e2[2] * q[2]) / det;
    if (t > 1e-7 && (!best || t < best.distance)) best = { index, distance: t };
  });

  return best as { index: number; distance: number } | null;
}

describe("SAH BVH", () => {
  const triangles = randomTriangles(600, 7);
  const bvh = new ShadowBvh(triangles, true), shadows = new ShadowBvh(triangles);
  const random = rng(99);

  // Half of the rays aim at the center of a random triangle, to guarantee plenty of hits; the other half are free.
  const rays = Array.from({ length: 300 }, (_, i) => {
    const origin = [random() * 24 - 12, random() * 24 - 12, random() * 24 - 12];
    let d = [random() - .5, random() - .5, random() - .5];

    if (i % 2 === 0) {
      const [a, b, c] = triangles[Math.floor(random() * triangles.length)].positions;
      d = [0, 1, 2].map(k => (a[k] + b[k] + c[k]) / 3 - origin[k] + (random() - .5) * .2);
    }

    const n = Math.hypot(d[0], d[1], d[2]);

    return { origin, d: d.map(v => v / n) };
  });

  it("returns the same closest hit as brute force", () => {
    let hits = 0;

    for (const { origin, d } of rays) {
      const expected = bruteForce(triangles, origin, d, false), hit = bvh.intersect(origin, d);
      if (expected) hits++;
      expect(hit?.index ?? null).toBe(expected?.index ?? null);
      if (hit && expected) expect(hit.distance).toBeCloseTo(expected.distance, 9);
    }

    expect(hits).toBeGreaterThan(60);
  });
  it("reports occlusion only for shadow casters and respects the range", () => {
    for (const { origin, d } of rays) {
      const expected = bruteForce(triangles, origin, d, true);
      expect(shadows.occluded(origin, d, Infinity, -1, () => true)).toBe(expected !== null);
      if (expected)
        expect(shadows.occluded(origin, d, expected.distance * .5, -1, () => true)).toBe(false);
    }
  });
  it("accepts axis-parallel rays and empty scenes", () => {
    expect(new ShadowBvh([], true).intersect([0, 0, 0], [0, 0, -1])).toBeNull();
    const wall = new ShadowBvh([{ positions: [[-1, -1, -2], [1, -1, -2], [0, 1, -2]], castShadow: true }]);
    expect(wall.occluded([0, 0, 0], [0, 0, -1], 3, -1, () => true)).toBe(true);
    expect(wall.occluded([5, 0, 0], [0, 0, -1], 3, -1, () => true)).toBe(false);
  });
});
