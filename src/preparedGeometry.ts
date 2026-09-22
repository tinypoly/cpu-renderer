import type { SerializedAttribute } from "./sceneSerialization.js";
import { buildBvh, type BvhData } from "./bvh.js";
import type { Value } from "./glsl.js";
import { cross, normalize, sub } from "./math.js";
import { sharedCopy, sharedFloat64, sharedUint32, sharedUint8 } from "./sharedBuffers.js";

/**
 * Shape of a flattened value. Number (0) and boolean (1) take one slot; a list interpolates item by item;
 * an object (matrix) is taken whole from the first vertex, as interpolation already did.
 */
export type ValueShape = 0 | 1 | ValueShape[] | { keys: [string, ValueShape][] };

export interface VertexLayout {
  attributes: [string, number][];
  varyings: [string, ValueShape][];
  /** Slots used per vertex, header included. */
  stride: number;
}

/** Header of each vertex: clip (4), world (3), normal (3) and point size (1). */
export const VERTEX_HEADER = 11;
export const CAST_SHADOW = 1;
export const RECEIVE_SHADOW = 2;
export const SHADOW_OPAQUE = 4;
/** No own normal or flat shading: the face normal is used. */
export const FLAT_NORMAL = 8;

export interface PreparedGroup {
  mesh: number;
  group: number;
  layout: VertexLayout;
}

/**
 * Geometry already processed for one camera and one resolution. The preparing worker writes it and the others only
 * read: everything lives in flat arrays, shared when the page is cross-origin isolated.
 */
export interface PreparedGeometry {
  groups: PreparedGroup[];
  vertices: Float64Array;
  /** Four per triangle: group and the offset of each vertex in `vertices`. */
  triangles: Uint32Array;
  flags: Uint8Array;
  /** Nine world coordinates per triangle, for the BVH and global illumination. */
  positions: Float64Array;
  faceNormals: Float64Array;
  /** Triangles of each bucket: `binTriangles[binOffsets[i]]` up to `binTriangles[binOffsets[i + 1]]`. */
  binOffsets: Uint32Array;
  binTriangles: Uint32Array;
  bvh: BvhData;
}

const mismatch = () => new Error("CPU renderer: a varying changed type between vertices.");

export function valueShape(value: Value): ValueShape {
  if (typeof value === "number")
    return 0;
  if (typeof value === "boolean")
    return 1;
  if (Array.isArray(value))
    return value.map(valueShape);
  if (value && typeof value === "object")
    return { keys: Object.entries(value).map(([key, child]) => [key, valueShape(child)]) };
  throw new Error("CPU renderer: varyings must be numeric.");
}

export function shapeSize(shape: ValueShape): number {
  if (shape === 0 || shape === 1)
    return 1;
  if (Array.isArray(shape))
    return shape.reduce((total: number, child) => total + shapeSize(child), 0);

  return shape.keys.reduce((total, [, child]) => total + shapeSize(child), 0);
}

/** Layout of a group, taken from the first vertex; points get the corners' `gl_PointCoord`. */
export function vertexLayout(
  attributes: Record<string, SerializedAttribute>, varyings: Record<string, Value>, points: boolean): VertexLayout {
  const attributeList = Object.entries(attributes)
    .map(([name, attribute]): [string, number] => [name, attribute.itemSize]);

  const varyingList = Object.entries(varyings).map(([name, value]): [string, ValueShape] => [name, valueShape(value)]);
  if (points)
    varyingList.push(["gl_PointCoord", [0, 0]]);

  const stride = VERTEX_HEADER + attributeList.reduce((total, [, size]) => total + size, 0)
    + varyingList.reduce((total, [, shape]) => total + shapeSize(shape), 0);

  return { attributes: attributeList, varyings: varyingList, stride };
}

/** Writes a value in the requested shape and returns the next slot. */
export function writeValue(value: Value | undefined, shape: ValueShape, out: Float64Array, offset: number): number {
  if (shape === 0) {
    if (typeof value !== "number")
      throw mismatch();
    out[offset] = value;

    return offset + 1;
  }

  if (shape === 1) {
    if (typeof value !== "boolean")
      throw mismatch();
    out[offset] = value ? 1 : 0;

    return offset + 1;
  }

  if (Array.isArray(shape)) {
    if (!Array.isArray(value) || value.length !== shape.length)
      throw mismatch();
    for (let i = 0; i < shape.length; i++)
      offset = writeValue(value[i], shape[i], out, offset);

    return offset;
  }

  if (!value || typeof value !== "object" || Array.isArray(value))
    throw mismatch();
  for (const [key, child] of shape.keys)
    offset = writeValue(value[key], child, out, offset);

  return offset;
}

/** Reads a value from a vertex; `cursor[0]` is the slot relative to `base` and advances. */
export function readValue(data: Float64Array, base: number, shape: ValueShape, cursor: number[]): Value {
  if (shape === 0)
    return data[base + cursor[0]++];
  if (shape === 1)
    return data[base + cursor[0]++] !== 0;
  if (Array.isArray(shape))
    return shape.map(child => readValue(data, base, child, cursor));

  return Object.fromEntries(shape.keys.map(([key, child]) => [key, readValue(data, base, child, cursor)]));
}

/** Interpolates a value across three stored vertices, following the same rule as object-based interpolation. */
export function interpolateStored(data: Float64Array, a: number, b: number, c: number, shape: ValueShape,
  w0: number, w1: number, w2: number, cursor: number[]): Value {
  if (shape === 0) {
    const k = cursor[0]++;

    return data[a + k] * w0 + data[b + k] * w1 + data[c + k] * w2;
  }

  if (shape === 1)
    return data[a + cursor[0]++] !== 0;
  if (Array.isArray(shape))
    return shape.map(child => interpolateStored(data, a, b, c, child, w0, w1, w2, cursor));

  return readValue(data, a, shape, cursor);
}

/** Builds geometry in growable arrays and copies to shared memory only at the end. */
export class GeometryWriter {
  vertices = new Float64Array(4096);
  readonly groups: PreparedGroup[] = [];
  private vertexLength = 0;
  private triangleData = new Uint32Array(4096);
  private flagData = new Uint8Array(1024);
  private triangleCount = 0;

  /** Reserves a vertex and returns its offset; `vertices` may be replaced by a larger array. */
  allocateVertex(stride: number): number {
    const next = this.vertexLength + stride;
    if (next > 0xffffffff)
      throw new Error("CPU renderer: scene geometry is too large.");

    if (next > this.vertices.length) {
      const grown = new Float64Array(Math.max(this.vertices.length * 2, next));
      grown.set(this.vertices.subarray(0, this.vertexLength));
      this.vertices = grown;
    }

    const offset = this.vertexLength;
    this.vertexLength = next;

    return offset;
  }

  addTriangle(group: number, a: number, b: number, c: number, flags: number): number {
    const index = this.triangleCount;

    if ((index + 1) * 4 > this.triangleData.length) {
      const triangles = new Uint32Array(this.triangleData.length * 2);
      const grownFlags = new Uint8Array(this.flagData.length * 2);
      triangles.set(this.triangleData);
      grownFlags.set(this.flagData);
      this.triangleData = triangles;
      this.flagData = grownFlags;
    }

    const o = index * 4;
    this.triangleData[o] = group;
    this.triangleData[o + 1] = a;
    this.triangleData[o + 2] = b;
    this.triangleData[o + 3] = c;
    this.flagData[index] = flags;
    this.triangleCount++;

    return index;
  }

  finish(bins: number[][], includeNonCasters: boolean): PreparedGeometry {
    const count = this.triangleCount;
    const vertices = sharedCopy(sharedFloat64, this.vertices.subarray(0, this.vertexLength));
    const triangles = sharedCopy(sharedUint32, this.triangleData.subarray(0, count * 4));
    const flags = sharedCopy(sharedUint8, this.flagData.subarray(0, count));
    // Release the growable staging arrays before allocating world triangles and the BVH.
    this.vertices = new Float64Array(0);
    this.triangleData = new Uint32Array(0);
    this.flagData = new Uint8Array(0);
    const positions = sharedFloat64(count * 9), faceNormals = sharedFloat64(count * 3);

    for (let t = 0; t < count; t++) {
      const corners = [1, 2, 3].map(k => {
        const o = triangles[t * 4 + k] + 4;

        return [vertices[o], vertices[o + 1], vertices[o + 2]];
      });

      corners.forEach((corner, k) => positions.set(corner, t * 9 + k * 3));
      faceNormals.set(normalize(cross(sub(corners[1], corners[0]), sub(corners[2], corners[0]))), t * 3);
    }

    const binOffsets = sharedUint32(bins.length + 1);
    let total = 0;
    bins.forEach((bin, index) => {
      binOffsets[index] = total;
      total += bin.length;
    });
    binOffsets[bins.length] = total;
    const binTriangles = sharedUint32(total);
    bins.forEach((bin, index) => {
      binTriangles.set(bin, binOffsets[index]);
      bin.length = 0;
    });
    const bvh = buildBvh(positions, count, index => includeNonCasters || (flags[index] & CAST_SHADOW) !== 0);

    return { groups: this.groups, vertices, triangles, flags, positions, faceNormals, binOffsets, binTriangles, bvh };
  }
}
