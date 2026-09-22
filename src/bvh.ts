import { sharedCopy, sharedFloat64, sharedInt32, sharedUint32 } from "./sharedBuffers.js";

export interface ShadowTriangle {
  positions: number[][];
  castShadow: boolean;
}

export interface RayHit {
  index: number;
  u: number;
  v: number;
  distance: number;
}

/** Tree in flat arrays: fits in shared memory and all workers traverse the same one. */
export interface BvhData {
  /** Minimum xyz and maximum xyz of each node. */
  bounds: Float64Array;
  /** Interior node: indices of both children. Leaf: `-1 - start` (its first entry in `indices`) and the count. */
  nodes: Int32Array;
  indices: Uint32Array;
}

export interface BvhSource {
  data: BvhData;
  /** Nine coordinates per triangle. */
  positions: Float64Array;
  /** Bit 1 set: the triangle casts direct shadows. */
  castShadow: Uint8Array;
}

/**
 * Binned SAH build (12 bins per axis, all three axes evaluated), up to 4 triangles per leaf;
 * the median split is used only when SAH cannot separate them.
 */
export function buildBvh(positions: Float64Array, count: number, include: (index: number) => boolean): BvhData {
  const bounds: number[] = [], nodes: number[] = [], leaves: number[] = [];
  const centroids = new Float64Array(count * 3);
  for (let i = 0; i < count; i++)
    for (let c = 0; c < 3; c++)
      centroids[i * 3 + c] = (positions[i * 9 + c] + positions[i * 9 + 3 + c] + positions[i * 9 + 6 + c]) / 3;

  const BINS = 32, binMin = new Float64Array(BINS * 3), binMax = new Float64Array(BINS * 3),
    binCount = new Int32Array(BINS);

  const leftArea = new Float64Array(BINS), leftCount = new Int32Array(BINS);

  const area = (min: number[], max: number[]) => {
    const x = max[0] - min[0], y = max[1] - min[1], z = max[2] - min[2];

    return 2 * (x * y + y * z + z * x);
  };

  const build = (indices: number[]): number => {
    const node = nodes.length / 2;
    const min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
    const cmin = [Infinity, Infinity, Infinity], cmax = [-Infinity, -Infinity, -Infinity];

    for (const i of indices) {
      for (let k = 0; k < 9; k++) {
        const c = k % 3, value = positions[i * 9 + k];
        if (value < min[c]) min[c] = value;
        if (value > max[c]) max[c] = value;
      }

      for (let c = 0; c < 3; c++) {
        const value = centroids[i * 3 + c];
        if (value < cmin[c]) cmin[c] = value;
        if (value > cmax[c]) cmax[c] = value;
      }
    }

    bounds.push(min[0], min[1], min[2], max[0], max[1], max[2]);
    nodes.push(0, 0);

    const leaf = () => {
      nodes[node * 2] = -1 - leaves.length;
      nodes[node * 2 + 1] = indices.length;
      leaves.push(...indices);

      return node;
    };

    if (indices.length <= 4)
      return leaf();
    const extent = [cmax[0] - cmin[0], cmax[1] - cmin[1], cmax[2] - cmin[2]];
    let left: number[], right: number[];

    if (extent[0] <= 0 && extent[1] <= 0 && extent[2] <= 0) {
      const mid = indices.length >> 1;
      left = indices.slice(0, mid); right = indices.slice(mid);
    } else {
      // All three axes are evaluated: the cheapest split is not always on the longest axis, and
      // testing the other two only costs build time, which is negligible here next to traversal.
      const parentArea = area(min, max);
      let bestCost = Infinity, bestSplit = -1, bestAxis = -1, bestScale = 0;

      for (let axis = 0; axis < 3; axis++) {
        if (extent[axis] <= 0) continue;
        binCount.fill(0); binMin.fill(Infinity); binMax.fill(-Infinity);
        const scale = BINS / extent[axis];
        const binOf = (i: number) => Math.min(BINS - 1, Math.floor((centroids[i * 3 + axis] - cmin[axis]) * scale));

        for (const i of indices) {
          const b = binOf(i);
          binCount[b]++;

          for (let k = 0; k < 9; k++) {
            const c = k % 3, value = positions[i * 9 + k];
            if (value < binMin[b * 3 + c]) binMin[b * 3 + c] = value;
            if (value > binMax[b * 3 + c]) binMax[b * 3 + c] = value;
          }
        }

        // Cost of each split between bins: left area times count, plus the same on the right.
        const rmin = [Infinity, Infinity, Infinity], rmax = [-Infinity, -Infinity, -Infinity];
        let n = 0;

        for (let b = 0; b < BINS; b++) {
          for (let c = 0; c < 3; c++) {
            rmin[c] = Math.min(rmin[c], binMin[b * 3 + c]); rmax[c] = Math.max(rmax[c], binMax[b * 3 + c]);
          }

          n += binCount[b];
          leftArea[b] = n ? area(rmin, rmax) : 0; leftCount[b] = n;
        }

        rmin.fill(Infinity); rmax.fill(-Infinity); n = 0;

        for (let b = BINS - 1; b > 0; b--) {
          for (let c = 0; c < 3; c++) {
            rmin[c] = Math.min(rmin[c], binMin[b * 3 + c]); rmax[c] = Math.max(rmax[c], binMax[b * 3 + c]);
          }

          n += binCount[b];
          if (!n || !leftCount[b - 1]) continue;
          const cost = leftArea[b - 1] * leftCount[b - 1] + area(rmin, rmax) * n;

          if (cost < bestCost) {
            bestCost = cost; bestSplit = b; bestAxis = axis; bestScale = scale;
          }
        }
      }

      if (bestSplit < 0 || (indices.length <= 8 && 1 + bestCost / parentArea >= indices.length))
        return leaf();

      const splitOf = (i: number) =>
        Math.min(BINS - 1, Math.floor((centroids[i * 3 + bestAxis] - cmin[bestAxis]) * bestScale));

      left = []; right = [];
      for (const i of indices)
        (splitOf(i) < bestSplit ? left : right).push(i);
    }

    const l = build(left), r = build(right);
    nodes[node * 2] = l;
    nodes[node * 2 + 1] = r;

    return node;
  };

  const included: number[] = [];
  for (let i = 0; i < count; i++)
    if (include(i))
      included.push(i);
  if (included.length)
    build(included);

  return {
    bounds: sharedCopy(sharedFloat64, bounds),
    nodes: sharedCopy(sharedInt32, nodes),
    indices: sharedCopy(sharedUint32, leaves),
  };
}

/** Ray entry distance into the node box, or Infinity when it does not cross within [0, far]. */
function slabEntry(bounds: Float64Array, b: number, ox: number, oy: number, oz: number,
  dx: number, dy: number, dz: number, ix: number, iy: number, iz: number, far: number): number {
  // A parallel ray outside the slab is rejected, including the edge-case NaNs.
  if ((dx === 0 && (ox < bounds[b] || ox > bounds[b + 3]))
    || (dy === 0 && (oy < bounds[b + 1] || oy > bounds[b + 4]))
    || (dz === 0 && (oz < bounds[b + 2] || oz > bounds[b + 5]))) return Infinity;
  let near = 0;
  let a = (bounds[b] - ox) * ix, c = (bounds[b + 3] - ox) * ix;

  if (a > c) { const t = a; a = c; c = t; }

  near = a > near ? a : near; far = c < far ? c : far;
  a = (bounds[b + 1] - oy) * iy; c = (bounds[b + 4] - oy) * iy;

  if (a > c) { const t = a; a = c; c = t; }

  near = a > near ? a : near; far = c < far ? c : far;
  a = (bounds[b + 2] - oz) * iz; c = (bounds[b + 5] - oz) * iz;

  if (a > c) { const t = a; a = c; c = t; }

  near = a > near ? a : near; far = c < far ? c : far;

  // NaN (zero direction on an axis with the origin inside the slab) must not discard the node.
  return near > far ? Infinity : near;
}

/** Immutable median-split BVH, shared by all buckets of a scene revision. */
export class ShadowBvh {
  private data: BvhData;
  private positions: Float64Array;
  private castShadow: Uint8Array;
  private stack: number[] = [];

  constructor(source: ShadowTriangle[] | BvhSource, includeNonCasters = false) {
    if (Array.isArray(source)) {
      this.positions = new Float64Array(source.length * 9);
      source.forEach((triangle, index) => this.positions.set(triangle.positions.flat(), index * 9));
      this.castShadow = Uint8Array.from(source, triangle => triangle.castShadow ? 1 : 0);
      this.data = buildBvh(this.positions, source.length, index => includeNonCasters || source[index].castShadow);
    } else {
      this.data = source.data;
      this.positions = source.positions;
      this.castShadow = source.castShadow;
    }
  }

  /** Iterative scalar traversal; returns immediately on the first accepted shadow hit. */
  occluded(
    origin: number[],
    direction: number[],
    distance: number,
    ignore: number,
    accepts: (index: number, u: number, v: number) => boolean): boolean {
    return this.trace(origin, direction, distance, ignore, accepts, false) !== null;
  }

  /** Closest scene hit, including surfaces that do not cast direct-light shadows. */
  intersect(origin: number[], direction: number[], distance = Infinity, ignore = -1): RayHit | null {
    return this.trace(origin, direction, distance, ignore, undefined, true);
  }

  private stackT: number[] = [];

  private trace(origin: number[], direction: number[], distance: number, ignore: number,
    accepts: ((index: number, u: number, v: number) => boolean) | undefined, closest: boolean): RayHit | null {
    const { bounds, nodes, indices } = this.data, positions = this.positions, castShadow = this.castShadow;
    if (!nodes.length)
      return null;
    let hit: RayHit | null = null;
    const epsilon = closest ? 1e-7 : 1e-5;
    const ox = origin[0], oy = origin[1], oz = origin[2], dx = direction[0], dy = direction[1], dz = direction[2];
    const ix = 1 / dx, iy = 1 / dy, iz = 1 / dz;
    const stack = this.stack, stackT = this.stackT;
    if (slabEntry(bounds, 0, ox, oy, oz, dx, dy, dz, ix, iy, iz, distance) === Infinity)
      return null;
    let top = 0;
    stack[top] = 0; stackT[top++] = 0;

    while (top > 0) {
      const node = stack[--top];
      // A node pushed before a closer hit is no longer relevant.
      if (stackT[top] >= distance)
        continue;
      const first = nodes[node * 2], second = nodes[node * 2 + 1];

      if (first >= 0) {
        // Nearer child on top of the stack: the closest hit shortens the ray for the rest.
        const t1 = slabEntry(bounds, first * 6, ox, oy, oz, dx, dy, dz, ix, iy, iz, distance);
        const t2 = slabEntry(bounds, second * 6, ox, oy, oz, dx, dy, dz, ix, iy, iz, distance);

        if (t1 <= t2) {
          if (t2 !== Infinity) { stack[top] = second; stackT[top++] = t2; }

          if (t1 !== Infinity) { stack[top] = first; stackT[top++] = t1; }
        } else {
          stack[top] = first; stackT[top++] = t1;
          stack[top] = second; stackT[top++] = t2;
        }

        continue;
      }

      for (let k = -1 - first, end = k + second; k < end; k++) {
        const index = indices[k];
        if (index === ignore || (!closest && !(castShadow[index] & 1)))
          continue;
        const o = index * 9, p0x = positions[o], p0y = positions[o + 1], p0z = positions[o + 2];
        const e1x = positions[o + 3] - p0x, e1y = positions[o + 4] - p0y, e1z = positions[o + 5] - p0z;
        const e2x = positions[o + 6] - p0x, e2y = positions[o + 7] - p0y, e2z = positions[o + 8] - p0z;
        const hx = dy * e2z - dz * e2y, hy = dz * e2x - dx * e2z, hz = dx * e2y - dy * e2x;
        const determinant = e1x * hx + e1y * hy + e1z * hz;
        if (Math.abs(determinant) < 1e-10)
          continue;
        const inverse = 1 / determinant;
        const sx = ox - p0x, sy = oy - p0y, sz = oz - p0z;
        const u = (sx * hx + sy * hy + sz * hz) * inverse;
        if (u < 0 || u > 1)
          continue;
        const qx = sy * e1z - sz * e1y, qy = sz * e1x - sx * e1z, qz = sx * e1y - sy * e1x;
        const v = (dx * qx + dy * qy + dz * qz) * inverse;
        if (v < 0 || u + v > 1)
          continue;
        const t = (e2x * qx + e2y * qy + e2z * qz) * inverse;

        if (t > epsilon && t < distance - epsilon && (!accepts || accepts(index, u, v))) {
          hit = { index, u, v, distance: t };
          if (!closest) return hit;
          distance = t;
        }
      }
    }

    return hit;
  }
}
