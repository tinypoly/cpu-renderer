import { describe, expect, it } from "vitest";
import { CpuComputePipeline, computeConfigKey, floatTexture, readPath, resolveBindings } from "./compute.js";
import type { GpgpuPassSpec, GpgpuPipelineDescriptor } from "./shaderTypes.js";

function drain(job: Generator<void>) {
  while (!job.next().done) { /* runs to completion */ }
}

/** Red channel of every texel of a pipeline output, row by row. */
function red(pipeline: CpuComputePipeline, uniform: string): number[] {
  const texture = pipeline.outputs()[uniform];

  return Array.from({ length: texture.width * texture.height }, (_, i) => texture.data[i * 4]);
}

/** One ping-pong buffer that each iteration increments by one, and records the iteration index in green. */
function counter(options: Partial<GpgpuPassSpec> = {}): GpgpuPipelineDescriptor {
  return {
    buffers: [{ name: "state", width: 2, height: 2, pingPong: true }],
    passes: [{
      name: "count",
      inputs: ["state"],
      output: "state",
      trigger: "everyFrame",
      fragmentShader: `void main() {
        vec4 previous = texture2D(state, vUv);
        gl_FragColor = vec4(previous.x + 1.0, float(uIteration), 0.0, 1.0);
      }`,
      ...options,
    }],
    outputs: { uState: "state" },
  };
}

describe("readPath", () => {
  it("follows dot paths and stops at missing or non-object values", () => {
    const data = { water: { size: 64, cascades: [{ scale: 2 }] }, label: "x" };
    expect(readPath(data, "water.size")).toBe(64);
    expect(readPath(data, "water.cascades.0.scale")).toBe(2);
    expect(readPath(data, "water.depth")).toBeUndefined();
    expect(readPath(data, "label.length")).toBeUndefined();
    expect(readPath(null, "water")).toBeUndefined();
  });
});

describe("resolveBindings", () => {
  it("converts each uniform type and falls back to the default", () => {
    const data = { tint: "#ff0000", on: 1, wind: [1, 2], amount: 0.25, count: "3" };
    expect(resolveBindings([
      { name: "uTint", type: "color", path: "tint", default: "#ffffff" },
      { name: "uWhite", type: "color", default: "#ffffff" },
      { name: "uOn", type: "bool", path: "on", default: false },
      { name: "uWind", type: "vec3", path: "wind", default: [0, 0, 0] },
      { name: "uFlat", type: "vec2", path: "amount", default: [0, 0] },
      { name: "uAmount", type: "float", path: "amount", default: 1 },
      { name: "uMissing", type: "float", path: "nothing", default: 0.5 },
      { name: "uCount", type: "int", path: "count", default: 0 },
    ], data)).toEqual({
      uTint: [1, 0, 0],
      uWhite: [1, 1, 1],
      uOn: true,
      // Missing components read as zero; a non-array value gives a zero vector.
      uWind: [1, 2, 0],
      uFlat: [0, 0],
      uAmount: 0.25,
      uMissing: 0.5,
      // Only numbers pass through: a numeric string is not converted.
      uCount: 0,
    });
  });

  it("builds arrays from per-element paths, object components and per-element defaults", () => {
    const data = { points: [{ x: 1, y: 2 }], first: 7 };
    expect(resolveBindings([
      { name: "uPoints", type: "vec2", arrayLength: 2, path: "points", components: ["x", "y"], default: [[0, 0], [9, 9]] },
      { name: "uValues", type: "float", arrayLength: 2, paths: ["first"], default: [0, 4] },
    ], data)).toEqual({ uPoints: [[1, 2], [9, 9]], uValues: [7, 4] });
  });

  it("rejects texture bindings, which must reference pipeline buffers", () => {
    expect(() => resolveBindings([{ name: "uMap", type: "texture", default: 0 }], {})).toThrow("pipeline buffers");
  });
});

describe("floatTexture", () => {
  it("allocates RGBA floats and filters outputs, which materials sample, but not intermediate buffers", () => {
    const output = floatTexture("a", 3, 2, true), buffer = floatTexture("b", 3, 2);
    expect(output.data).toHaveLength(3 * 2 * 4);
    expect(output.nearest).toBe(false);
    expect(buffer.nearest).toBe(true);
    expect(output.wrapS).not.toBe(buffer.wrapS);
  });

  it.each([[0, 4], [2.5, 4], [4096, 1025]])("rejects a %ix%i buffer", (width, height) => {
    expect(() => floatTexture("x", width, height)).toThrow("invalid buffer resolution");
  });
});

describe("computeConfigKey", () => {
  const descriptor: GpgpuPipelineDescriptor = {
    buffers: [{ name: "height", width: { path: "size", default: 8 }, height: 4 }],
    passes: [{
      name: "fill", inputs: [], output: "height", trigger: "onConfigChange",
      fragmentShader: "void main() { gl_FragColor = vec4(uAmount); }",
      uniforms: [
        { name: "uAmount", type: "float", path: "amount", default: 0 },
        { name: "uSource", type: "texture", default: 0 },
      ],
      iterations: { path: "size", default: 8, transform: "log2" },
    }],
    outputs: { uHeight: "height" },
  };

  it("changes with the uniform data and the buffer sizes, and ignores unrelated data", () => {
    const key = computeConfigKey(descriptor, { size: 16, amount: 1 });
    expect(computeConfigKey(descriptor, { size: 16, amount: 1, label: "other" })).toBe(key);
    expect(computeConfigKey(descriptor, { size: 16, amount: 2 })).not.toBe(key);
    expect(computeConfigKey(descriptor, { size: 32, amount: 1 })).not.toBe(key);
    // Missing paths fall back to the defaults.
    expect(computeConfigKey(descriptor, { amount: 1 })).toBe(computeConfigKey(descriptor, { size: 8, amount: 1 }));
  });
});

describe("CpuComputePipeline", () => {
  it("runs a pass over every texel with vUv, gl_FragCoord and the bound uniforms", () => {
    const pipeline = new CpuComputePipeline({
      buffers: [{ name: "field", width: 2, height: 2 }],
      passes: [{
        name: "fill", inputs: [], output: "field", trigger: "everyFrame",
        uniforms: [{ name: "uScale", type: "float", path: "scale", default: 1 }],
        fragmentShader: "void main() { gl_FragColor = vec4(vUv.x * uScale, gl_FragCoord.y, uTime, uDeltaTime); }",
      }],
      outputs: { uField: "field" },
    }, "sim", { scale: 4 });

    drain(pipeline.tick(3, 0.5));
    const { data, id } = pipeline.outputs().uField;
    expect(id).toBe("sim:field");
    expect(Array.from(data)).toEqual([
      1, 0.5, 3, 0.5, 3, 0.5, 3, 0.5,
      1, 1.5, 3, 0.5, 3, 1.5, 3, 0.5,
    ]);
  });

  it("alternates ping-pong slots across iterations, including an iteration count derived with log2", () => {
    const pipeline = new CpuComputePipeline(counter({ iterations: { path: "size", default: 2, transform: "log2" } }),
      "sim", { size: 8 });

    drain(pipeline.tick(0));
    const texture = pipeline.outputs().uState;
    // Three iterations: each reads the previous result; the last one wrote uIteration = 2.
    expect(red(pipeline, "uState")).toEqual([3, 3, 3, 3]);
    expect(texture.data[1]).toBe(2);
  });

  it("reruns everyFrame passes on each tick and onConfigChange passes only when their uniforms change", () => {
    const everyFrame = new CpuComputePipeline(counter(), "a", {});
    drain(everyFrame.tick(0));
    drain(everyFrame.tick(1));
    expect(red(everyFrame, "uState")).toEqual([2, 2, 2, 2]);

    const data = { amount: 1 };

    const onChange = new CpuComputePipeline(counter({
      trigger: "onConfigChange", uniforms: [{ name: "uAmount", type: "float", path: "amount", default: 0 }],
    }), "b", data);

    drain(onChange.tick(0));
    drain(onChange.tick(1));
    expect(red(onChange, "uState")).toEqual([1, 1, 1, 1]);
    data.amount = 2;
    drain(onChange.tick(2));
    expect(red(onChange, "uState")).toEqual([2, 2, 2, 2]);
  });

  it("yields once per row so a long bake can be interrupted", () => {
    const pipeline = new CpuComputePipeline(counter({ iterations: 2 }), "sim", {});
    const job = pipeline.tick(0);
    let yields = 0;
    while (!job.next().done)
      yields++;
    expect(yields).toBe(2 * 2);
  });

  it("restores outputs from a base64 or float snapshot instead of simulating", () => {
    const values = Float32Array.from({ length: 16 }, (_, i) => i / 2);
    const encoded = btoa(String.fromCharCode(...new Uint8Array(values.buffer)));

    for (const data of [encoded, values]) {
      const pipeline = new CpuComputePipeline(counter(), "sim", {});
      pipeline.restore({ outputs: { uState: { width: 2, height: 2, data } } });
      expect(Array.from(pipeline.outputs().uState.data)).toEqual(Array.from(values));
    }
  });

  it("rejects a snapshot that is missing an output or does not match its size", () => {
    const pipeline = new CpuComputePipeline(counter(), "sim", {});
    expect(() => pipeline.restore({ outputs: {} })).toThrow("missing uState");
    expect(() => pipeline.restore({ outputs: { uState: { width: 4, height: 1, data: new Float32Array(16) } } }))
      .toThrow("resolution differs");
    expect(() => pipeline.restore({ outputs: { uState: { width: 2, height: 2, data: new Float32Array(3) } } }))
      .toThrow("invalid snapshot");
  });

  it("rejects descriptors whose buffers or passes do not line up", () => {
    const base = counter();
    expect(() => new CpuComputePipeline({ ...base, buffers: [...base.buffers, ...base.buffers] }, "sim", {}))
      .toThrow("duplicate buffer state");
    expect(() => new CpuComputePipeline(counter({ output: "missing" }), "sim", {})).toThrow("invalid buffer binding");
    expect(() => new CpuComputePipeline(counter({ inputs: ["missing"] }), "sim", {})).toThrow("invalid buffer binding");
    expect(() => new CpuComputePipeline({ ...base, buffers: [{ name: "state", width: 2, height: 2 }] }, "sim", {}))
      .toThrow("requires a ping-pong buffer");
    expect(() => new CpuComputePipeline({ ...base, outputs: { uOther: "missing" } }, "sim", {}).outputs())
      .toThrow("unknown output missing");
  });

  it("stops on an iteration count out of range and on non-finite pixels", () => {
    expect(() => drain(new CpuComputePipeline(counter({ iterations: 5000 }), "sim", {}).tick(0)))
      .toThrow("invalid iteration count");

    const infinite = new CpuComputePipeline({
      buffers: [{ name: "field", width: 1, height: 1 }],
      passes: [{ name: "broken", inputs: [], output: "field", trigger: "everyFrame",
        fragmentShader: "void main() { gl_FragColor = vec4(1.0 / 0.0); }" }],
      outputs: { uField: "field" },
    }, "sim", {});

    expect(() => drain(infinite.tick(0))).toThrow("broken produced non-finite pixels at 0,0");
  });
});
