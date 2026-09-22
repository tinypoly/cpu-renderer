/** Hands buckets on demand to ready workers, center-out; one bucket in flight per worker. */
export class BucketScheduler {
  private pending: number[] = [];
  private ready = new Set<number>();
  private active = new Map<number, number>();
  completed = 0;
  total = 0;

  /** Takes the bucket indices in the order they should be handed out. */
  reset(order: number[]) {
    this.pending = [...order];
    this.ready.clear();
    this.active.clear();
    this.completed = 0;
    this.total = order.length;
  }

  /** A worker finished preparing: returns its first bucket, if any remain. */
  markReady(worker: number): number | null {
    this.ready.add(worker);

    return this.next(worker);
  }

  /** Bucket done: counts it and returns the same worker's next one. Buckets from another assignment don't count. */
  complete(worker: number, bucket: number): number | null {
    if (this.active.get(worker) !== bucket)
      return null;
    this.active.delete(worker);
    this.completed++;

    return this.next(worker);
  }

  private next(worker: number): number | null {
    if (!this.ready.has(worker) || this.active.has(worker) || !this.pending.length)
      return null;
    const bucket = this.pending.shift()!;
    this.active.set(worker, bucket);

    return bucket;
  }

  get busy() {
    return this.pending.length > 0 || this.active.size > 0;
  }
}
