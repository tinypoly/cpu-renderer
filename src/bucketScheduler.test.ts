import { describe, expect, it } from "vitest";
import { BucketScheduler } from "./bucketScheduler.js";

describe("BucketScheduler", () => {
  it("hands buckets out on demand, one in flight per ready worker", () => {
    const scheduler = new BucketScheduler();
    // Grid indices ordered center-out, as the worker announces on start.
    scheduler.reset([7, 6, 8, 2, 12]);
    expect(scheduler.markReady(0)).toBe(7);
    expect(scheduler.markReady(1)).toBe(6);
    // One bucket per worker: a worker with a bucket in flight gets no other, and a mismatched completion doesn't count.
    expect(scheduler.complete(0, 6)).toBeNull();
    expect(scheduler.complete(0, 7)).toBe(8);
    expect(scheduler.complete(1, 6)).toBe(2);
    expect(scheduler.complete(0, 8)).toBe(12);
    expect(scheduler.complete(1, 2)).toBeNull();
    expect(scheduler.busy).toBe(true);
    expect(scheduler.complete(0, 12)).toBeNull();
    expect(scheduler.completed).toBe(5);
    expect(scheduler.busy).toBe(false);
  });
  it("serves late workers from the remaining queue and ignores stale completions", () => {
    const scheduler = new BucketScheduler();
    scheduler.reset([0, 1, 2]);
    expect(scheduler.markReady(0)).toBe(0);
    expect(scheduler.complete(0, 0)).toBe(1);
    expect(scheduler.markReady(1)).toBe(2);
    scheduler.reset([0, 1]);
    // After a restart, buckets from the previous revision do not count and no worker is ready.
    expect(scheduler.complete(0, 1)).toBeNull();
    expect(scheduler.completed).toBe(0);
    expect(scheduler.markReady(1)).toBe(0);
  });
});
