import { sharedFloat32, sharedInt32 } from "./sharedBuffers.js";

/** A point where indirect light was actually computed, and how far that result remains valid. */
export interface IrradianceRecord {
  position: number[];
  normal: number[];
  color: number[];
  variance: number;
  /** Distance up to which the record still describes the indirect light; lookups weight by the fraction used. */
  radius: number;
}

export interface IrradianceLookup {
  color: number[];
  variance: number;
}

/** A record behind the plane of the queried point sees over a corner: beyond this tolerance it is rejected. */
const PLANE_TOLERANCE = .05;
/** Slots of a packed record: position (3), normal (3), color (3), variance and radius. */
export const RECORD_STRIDE = 11;

/**
 * Weight of a record for a point, from Ward's error: zero when the record does not apply (outside the radius, normal
 * too different or point behind the record's plane), otherwise larger the closer and more aligned it is. It fades to
 * zero at the edge of the record instead of Ward's 1/error: a record that still weighs one where it stops applying
 * leaves a visible step around every record (scales), and a weight peaked at the record keeps each one's noise.
 */
function recordWeight(position: number[], normal: number[], px: number, py: number, pz: number,
  nx: number, ny: number, nz: number, radius: number): number {
  const dx = position[0] - px, dy = position[1] - py, dz = position[2] - pz;
  const distance = Math.sqrt(dx * dx + dy * dy + dz * dz);
  if (distance >= radius)
    return 0;
  const cosine = normal[0] * nx + normal[1] * ny + normal[2] * nz;
  // Ward error: fraction of the radius covered plus the normal difference; above one the record does not apply.
  const error = distance / radius + Math.sqrt(Math.max(0, 1 - cosine));
  if (error >= 1)
    return 0;
  if ((dx * (nx + normal[0]) + dy * (ny + normal[1]) + dz * (nz + normal[2])) * .5 < -PLANE_TOLERANCE * radius)
    return 0;

  return (1 - error) * (1 - error);
}

/**
 * Summed weight below which the prepass still places a record: the weight of a single record at half its radius.
 * Placing only where no record reaches leaves most points interpolating one record, which shows the grid.
 */
export const PLACEMENT_COVERAGE = .25;

/**
 * How much farther a record reaches when shading looks it up than when the prepass places it. Each record carries
 * its own sampling error, independent of its neighbors', and interpolating few of them shows it as low-frequency
 * blotches that more samples barely reduce. Looking up a wider radius averages several times more records at each
 * point (the smoothing of a classic irradiance cache) without computing any more of them.
 */
export const LOOKUP_SMOOTHING = 3;

/**
 * How many grid cells a record can reach. It is also the radius of the lookup window, so each extra cell costs a
 * larger scan on every fragment; three already covers a flat floor with few records.
 */
export const GRID_REACH = 3;

/** Arrays of the full-image record grid; shared when the page allows it. */
export interface IrradianceGridBuffers {
  width: number;
  height: number;
  spacing: number;
  columns: number;
  rows: number;
  data: Float32Array;
  /** 1 when the cell has a record; written last, with Atomics, so the reader sees the complete record. */
  flags: Int32Array;
}

export function allocateIrradianceGrid(width: number, height: number, spacing: number): IrradianceGridBuffers {
  const columns = Math.ceil(width / spacing), rows = Math.ceil(height / spacing);

  return { width, height, spacing, columns, rows,
    data: sharedFloat32(columns * rows * RECORD_STRIDE), flags: sharedInt32(columns * rows) };
}

/**
 * Record grid for the full image, one cell per point of the finest grid. The cache pass writes it (each cell by a
 * single worker) and shading queries the window of cells around the pixel, which includes records from neighboring
 * buckets.
 */
export class IrradianceGrid {
  constructor(readonly buffers: IrradianceGridBuffers) {}

  cellX(px: number) {
    return Math.min(this.buffers.columns - 1, Math.max(0, Math.floor(px / this.buffers.spacing)));
  }

  cellY(py: number) {
    return Math.min(this.buffers.rows - 1, Math.max(0, Math.floor(py / this.buffers.spacing)));
  }

  has(cx: number, cy: number) {
    return Atomics.load(this.buffers.flags, cy * this.buffers.columns + cx) === 1;
  }

  write(cx: number, cy: number, record: IrradianceRecord) {
    const i = cy * this.buffers.columns + cx, o = i * RECORD_STRIDE, data = this.buffers.data;
    data.set(record.position, o); data.set(record.normal, o + 3); data.set(record.color, o + 6);
    data[o + 9] = record.variance; data[o + 10] = record.radius;
    Atomics.store(this.buffers.flags, i, 1);

    return i;
  }

  /**
   * Interpolates the records in the window around the pixel that reach the point; null when none does, or when
   * their summed weight does not exceed `coverage`. `smoothing` widens both the records and the window: placement
   * asks with 1, shading with `LOOKUP_SMOOTHING`.
   */
  lookupAt(px: number, py: number, position: number[], normal: number[], coverage = 0,
    smoothing = 1): IrradianceLookup | null {
    const { columns, rows, data, flags } = this.buffers;
    const cx = this.cellX(px), cy = this.cellY(py);
    const reach = Math.ceil(GRID_REACH * smoothing);
    const x0 = Math.max(0, cx - reach), x1 = Math.min(columns - 1, cx + reach);
    const y0 = Math.max(0, cy - reach), y1 = Math.min(rows - 1, cy + reach);
    let weightSum = 0, r = 0, g = 0, b = 0, variance = 0;

    for (let y = y0; y <= y1; y++)
      for (let x = x0; x <= x1; x++) {
        const i = y * columns + x;
        if (Atomics.load(flags, i) !== 1)
          continue;
        const o = i * RECORD_STRIDE;

        const weight = recordWeight(position, normal, data[o], data[o + 1], data[o + 2],
          data[o + 3], data[o + 4], data[o + 5], data[o + 10] * smoothing);

        if (weight === 0)
          continue;
        weightSum += weight;
        r += data[o + 6] * weight; g += data[o + 7] * weight; b += data[o + 8] * weight;
        variance += data[o + 9] * weight;
      }

    if (weightSum <= coverage)
      return null;

    return { color: [r / weightSum, g / weightSum, b / weightSum], variance: variance / weightSum };
  }

  /** Written cells, each record prefixed by its index, for transfer without shared memory. */
  exportCells(cells: number[]): Float32Array<ArrayBuffer> {
    const out = new Float32Array(cells.length * (RECORD_STRIDE + 1));
    cells.forEach((cell, k) => {
      out[k * (RECORD_STRIDE + 1)] = cell;
      out.set(
        this.buffers.data.subarray(cell * RECORD_STRIDE, (cell + 1) * RECORD_STRIDE),
        k * (RECORD_STRIDE + 1) + 1);
    });

    return out;
  }

  importCells(packed: Float32Array) {
    for (let k = 0; k + RECORD_STRIDE + 1 <= packed.length; k += RECORD_STRIDE + 1) {
      const cell = packed[k];
      this.buffers.data.set(packed.subarray(k + 1, k + 1 + RECORD_STRIDE), cell * RECORD_STRIDE);
      Atomics.store(this.buffers.flags, cell, 1);
    }
  }
}

/**
 * Irradiance cache for a bucket, in the spirit of Ward's: sparse records with a validity radius, and lookups
 * interpolate those that reach the point, weighting closer records with more similar normals higher.
 */
export class IrradianceCache {
  private records: IrradianceRecord[] = [];

  get size() {
    return this.records.length;
  }

  add(record: IrradianceRecord) {
    this.records.push(record);
  }

  /** Records in a flat array, to travel between the cache pass and the shading pass. */
  pack(): Float32Array<ArrayBuffer> {
    const out = new Float32Array(this.records.length * RECORD_STRIDE);
    this.records.forEach((record, i) => {
      const o = i * RECORD_STRIDE;
      out.set(record.position, o); out.set(record.normal, o + 3); out.set(record.color, o + 6);
      out[o + 9] = record.variance; out[o + 10] = record.radius;
    });

    return out;
  }

  static fromPacked(data: Float32Array): IrradianceCache {
    const cache = new IrradianceCache();
    for (let o = 0; o + RECORD_STRIDE <= data.length; o += RECORD_STRIDE)
      cache.records.push({
        position: [data[o], data[o + 1], data[o + 2]], normal: [data[o + 3], data[o + 4], data[o + 5]],
        color: [data[o + 6], data[o + 7], data[o + 8]], variance: data[o + 9], radius: data[o + 10],
      });

    return cache;
  }

  lookup(position: number[], normal: number[], smoothing = 1): IrradianceLookup | null {
    let weightSum = 0, r = 0, g = 0, b = 0, variance = 0;

    for (const record of this.records) {
      const weight = recordWeight(position, normal, record.position[0], record.position[1], record.position[2],
        record.normal[0], record.normal[1], record.normal[2], record.radius * smoothing);

      if (weight === 0)
        continue;
      weightSum += weight;
      r += record.color[0] * weight; g += record.color[1] * weight; b += record.color[2] * weight;
      variance += record.variance * weight;
    }

    if (weightSum === 0)
      return null;

    return { color: [r / weightSum, g / weightSum, b / weightSum], variance: variance / weightSum };
  }
}
