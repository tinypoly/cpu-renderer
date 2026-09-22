import { describe, expect, it } from "vitest";
import {
  allocateIrradianceGrid, IrradianceCache, IrradianceGrid, LOOKUP_SMOOTHING, PLACEMENT_COVERAGE, RECORD_STRIDE,
} from "./irradianceCache.js";

const up = [0, 1, 0];
describe("irradiance cache", () => {
  it("interpolates records that reach the point, weighting by distance", () => {
    const cache = new IrradianceCache();
    cache.add({ position: [0, 0, 0], normal: up, color: [1, 0, 0], variance: .1, radius: 2 });
    cache.add({ position: [2, 0, 0], normal: up, color: [0, 1, 0], variance: .3, radius: 2 });
    const middle = cache.lookup([1, 0, 0], up)!;
    expect(middle.color[0]).toBeCloseTo(.5, 6);
    expect(middle.color[1]).toBeCloseTo(.5, 6);
    expect(middle.variance).toBeCloseTo(.2, 6);
    const near = cache.lookup([.2, 0, 0], up)!;
    expect(near.color[0]).toBeGreaterThan(.8);
    expect(cache.lookup([0, 0, 0], up)!.color).toEqual([1, 0, 0]);
  });
  it("fades a record out at the edge of its radius, without a step", () => {
    const cache = new IrradianceCache();
    cache.add({ position: [0, 0, 0], normal: up, color: [1, 0, 0], variance: 0, radius: 4 });
    cache.add({ position: [1, 0, 0], normal: up, color: [0, 1, 0], variance: 0, radius: 1 });
    // Just inside the second record its contribution has already vanished, so leaving it is no jump.
    expect(cache.lookup([1.999, 0, 0], up)!.color[1]).toBeLessThan(.001);
    expect(cache.lookup([2.001, 0, 0], up)!.color).toEqual([1, 0, 0]);
    expect(cache.lookup([1.5, 0, 0], up)!.color[1]).toBeGreaterThan(.2);
  });
  it("rejects records outside the radius, with a different normal or behind the point's plane", () => {
    const cache = new IrradianceCache();
    cache.add({ position: [0, 0, 0], normal: up, color: [1, 1, 1], variance: 0, radius: 1 });
    expect(cache.lookup([1.5, 0, 0], up)).toBeNull();
    expect(cache.lookup([.2, 0, 0], [1, 0, 0])).toBeNull();
    // Point below the record's plane, like the step of a corner: the record would see over the edge.
    expect(cache.lookup([.3, -.5, 0], up)).toBeNull();
    expect(cache.lookup([.3, .01, 0], up)).not.toBeNull();
    expect(cache.size).toBe(1);
  });
});
describe("full-image record grid", () => {
  it("queries only the window around the pixel and travels packed", () => {
    const grid = new IrradianceGrid(allocateIrradianceGrid(64, 32, 4));
    expect(grid.buffers.columns).toBe(16);
    expect(grid.has(2, 3)).toBe(false);
    const cell = grid.write(2, 3, { position: [1, 0, 0], normal: up, color: [.5, .5, .5], variance: .01, radius: 3 });
    expect(grid.has(2, 3)).toBe(true);
    expect(grid.lookupAt(9, 13, [1.2, 0, 0], up)!.color[0]).toBeCloseTo(.5, 6);
    // Placement asks for more than a faint reach: near the edge of the record the point still needs its own.
    expect(grid.lookupAt(9, 13, [3.5, 0, 0], up)).not.toBeNull();
    expect(grid.lookupAt(9, 13, [3.5, 0, 0], up, PLACEMENT_COVERAGE)).toBeNull();
    expect(grid.lookupAt(9, 13, [1.2, 0, 0], up, PLACEMENT_COVERAGE)).not.toBeNull();
    // Shading looks records up wider than placement does, to average more of them.
    expect(grid.lookupAt(9, 13, [5, 0, 0], up)).toBeNull();
    expect(grid.lookupAt(9, 13, [5, 0, 0], up, 0, LOOKUP_SMOOTHING)!.color[0]).toBeCloseTo(.5, 6);
    // Same world point, but the pixel is too far from the cell: the window does not include it.
    expect(grid.lookupAt(60, 30, [1.2, 0, 0], up)).toBeNull();
    const packed = grid.exportCells([cell]);
    expect(packed.length).toBe(RECORD_STRIDE + 1);
    const copy = new IrradianceGrid(allocateIrradianceGrid(64, 32, 4));
    copy.importCells(packed);
    expect(copy.has(2, 3)).toBe(true);
    expect(copy.lookupAt(9, 13, [1, 0, 0], up)!.variance).toBeCloseTo(.01, 6);
  });
});
