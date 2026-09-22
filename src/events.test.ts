import { describe, expect, it } from "vitest";
import { EventEmitter } from "./events.js";

class Counter extends EventEmitter<{ tick: number; done: void }> {
  fire(value: number) {
    this.emit("tick", value);
  }

  finish() {
    this.emit("done");
  }
}

describe("EventEmitter", () => {
  it("delivers payloads, unsubscribes through the returned function and honours once", () => {
    const counter = new Counter();
    const ticks: number[] = [];
    let finished = 0;
    const off = counter.on("tick", value => ticks.push(value));
    counter.once("done", () => finished++);
    counter.fire(1);
    counter.finish();
    counter.finish();
    off();
    counter.fire(2);
    expect(ticks).toEqual([1]);
    expect(finished).toBe(1);
  });
  it("lets a listener remove another one during delivery", () => {
    const counter = new Counter();
    const seen: string[] = [];
    const second = () => seen.push("second");
    counter.on("tick", () => {
      seen.push("first");
      counter.off("tick", second);
    });
    counter.on("tick", second);
    counter.fire(0);
    counter.removeAllListeners();
    counter.fire(1);
    expect(seen).toEqual(["first"]);
  });
});
