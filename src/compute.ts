import { Color } from "three";
import type { GpgpuPipelineDescriptor, GpgpuSize, ShaderUniformBinding } from "./shaderTypes.js";
import type { SerializedTexture, UniformValue } from "./sceneSerialization.js";
import { CpuShader, type Value } from "./glsl.js";
import { sampleTexture, type TextureSampling } from "./texture.js";
import { sharedFloat32 } from "./sharedBuffers.js";

export const readPath = (data: unknown, path: string): unknown => path.split(".").reduce<unknown>(
  (value, key) => value && typeof value === "object" ? (value as Record<string, unknown>)[key] : undefined, data);

export function resolveBindings(bindings: ShaderUniformBinding[], data: unknown): Record<string, UniformValue> {
  return Object.fromEntries(bindings.map(binding => {
    const scalar = (raw: unknown, fallback: unknown): UniformValue => {
      const value = raw === undefined ? fallback : raw;
      if (binding.type === "color") return new Color((value as string | number | undefined) ?? "#ffffff").toArray();
      if (binding.type === "bool") return Boolean(value);
      if (binding.type.startsWith("vec")) return Array.from({ length: Number(binding.type.slice(3)) }, (_, i) =>
        Number(Array.isArray(value) ? value[i] ?? 0 : 0));
      if (binding.type === "texture") throw new Error("CPU compute: texture bindings must reference pipeline buffers.");

      return typeof value === "number" ? value : 0;
    };

    const raw = binding.path ? readPath(data, binding.path) : undefined;

    const value = binding.arrayLength ? Array.from({ length: binding.arrayLength }, (_, i) => {
      let entry = binding.paths?.[i] ? readPath(data, binding.paths[i]) : Array.isArray(raw) ? raw[i] : undefined;
      if (binding.components && entry && typeof entry === "object")
        entry = binding.components.map(key => Number((entry as Record<string, unknown>)[key] ?? 0));

      return scalar(entry, Array.isArray(binding.default) ? binding.default[i] : undefined);
    }) : scalar(raw, binding.default);

    return [binding.name, value];
  }));
}

/** Pipeline outputs computed elsewhere, as base64 or floats (read back from a GPU simulation, for example). */
export interface CpuComputeSnapshot {
  version?: number;
  outputs: Record<string, { width: number; height: number; data: string | Float32Array }>;
}

function decodeFloats(base64: string): Float32Array {
  const binary = atob(base64), bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);

  return new Float32Array(bytes.buffer);
}

export function floatTexture(id: string, width: number, height: number, output = false): SerializedTexture {
  if (!Number.isInteger(width) || !Number.isInteger(height) || width < 1 || height < 1 || width * height > 4194304)
    throw new Error("CPU compute: invalid buffer resolution (maximum 4 megapixels).");

  return { id, width, height, data: sharedFloat32(width * height * 4), colorSpace: "",
    wrapS: output ? 1000 : 1001, wrapT: output ? 1000 : 1001, nearest: !output,
    matrix: [1, 0, 0, 0, 1, 0, 0, 0, 1], flipY: false, channel: 0 };
}

function size(spec: GpgpuSize, data: unknown) {
  const raw = typeof spec === "number" ? spec : readPath(data, spec.path) ?? spec.default;

  return Math.max(1, Math.round(typeof spec === "object" && spec.transform === "log2" ? Math.log2(Number(raw)) : Number(raw)));
}

/** Identifies the data feeding the passes; while it stays the same, already computed outputs remain valid. */
export function computeConfigKey(descriptor: GpgpuPipelineDescriptor, data: unknown): string {
  return JSON.stringify([
    descriptor.buffers.map(buffer => [size(buffer.width, data), size(buffer.height, data)]),
    descriptor.passes.map(pass => [
      resolveBindings((pass.uniforms ?? []).filter(binding => binding.type !== "texture"), data),
      pass.iterations === undefined ? 1 : size(pass.iterations, data),
    ]),
  ]);
}

/** Executes fragment passes on float buffers, on the CPU. No renderer or GL context. */
export class CpuComputePipeline {
  private buffers = new Map<string, { front: SerializedTexture; back: SerializedTexture }>();
  private config = new Map<string, string>();
  private programs: CpuShader[];

  constructor(readonly descriptor: GpgpuPipelineDescriptor, readonly id: string, readonly data: unknown) {
    const outputs = new Set(Object.values(descriptor.outputs));

    for (const buffer of descriptor.buffers) {
      if (this.buffers.has(buffer.name)) throw new Error(`CPU compute: duplicate buffer ${buffer.name}.`);
      const texture = floatTexture(`${id}:${buffer.name}`, size(buffer.width, data), size(buffer.height, data), outputs.has(buffer.name));
      this.buffers.set(buffer.name, {
        front: texture,
        back: buffer.pingPong ? { ...texture, data: new Float32Array(texture.data.length) } : texture,
      });
    }

    this.programs = descriptor.passes.map(pass => {
      if (!this.buffers.has(pass.output) || pass.inputs.some(input => !this.buffers.has(input)))
        throw new Error(`CPU compute: invalid buffer binding in ${pass.name}.`);
      if (pass.inputs.includes(pass.output) && !descriptor.buffers.find(b => b.name === pass.output)?.pingPong)
        throw new Error(`CPU compute: ${pass.name} requires a ping-pong buffer.`);

      const declarations = (pass.uniforms ?? []).map(b => {
        const type = b.type === "color" ? "vec3" : b.type === "texture" ? "sampler2D" : b.type;

        return `uniform ${type} ${b.name}${b.arrayLength ? `[${b.arrayLength}]` : ""};`;
      });

      const samplers = pass.inputs.map(n => `uniform sampler2D ${n};`).join("\n");

      return new CpuShader("varying vec2 vUv; uniform int uIteration; uniform float uTime; uniform float uDeltaTime;\n"
        + samplers + "\n" + declarations.join("\n") + "\n" + pass.fragmentShader);
    });
  }

  restore(snapshot: CpuComputeSnapshot) {
    for (const [uniform, name] of Object.entries(this.descriptor.outputs)) {
      const output = snapshot.outputs[uniform];
      if (!output) throw new Error(`CPU compute: paused snapshot is missing ${uniform}.`);
      const state = this.buffers.get(name)!;
      if (state.front.width !== output.width || state.front.height !== output.height)
        throw new Error(`CPU compute: paused snapshot resolution differs for ${uniform}.`);
      const values = typeof output.data === "string" ? decodeFloats(output.data) : output.data;
      if (values.length !== state.front.data.length) throw new Error(`CPU compute: invalid snapshot ${uniform}.`);
      state.front.data.set(values);
    }
  }

  *tick(time: number, deltaTime = 0): Generator<void> {
    for (let index = 0; index < this.descriptor.passes.length; index++) {
      const pass = this.descriptor.passes[index], uniforms = resolveBindings(pass.uniforms ?? [], this.data);
      const config = JSON.stringify(uniforms);
      if (pass.trigger === "onConfigChange" && this.config.get(pass.name) === config) continue;
      const state = this.buffers.get(pass.output)!;
      const iterations = pass.iterations === undefined ? 1 : size(pass.iterations, this.data);
      if (!Number.isInteger(iterations) || iterations > 4096)
        throw new Error(`CPU compute: invalid iteration count in ${pass.name}.`);

      for (let iteration = 0; iteration < iterations; iteration++) {
        const inputs = new Map(pass.inputs.map(name => [name, this.buffers.get(name)!.front]));

        const context = { texture: (sampler: Value, uv: number[], sampling?: TextureSampling) => {
          const texture = inputs.get(String(sampler));
          if (!texture) throw new Error(`CPU compute: unbound sampler in ${pass.name}.`);

          return sampleTexture(texture, uv, false, sampling);
        } };

        const globals = {
          ...uniforms, ...Object.fromEntries(pass.inputs.map(n => [n, n])),
          uTime: time, uDeltaTime: deltaTime, uIteration: iteration,
        };

        const target = state.back;

        for (let y = 0; y < target.height; y++) {
          for (let x = 0; x < target.width; x++) {
            const result = this.programs[index].run({
              ...globals, vUv: [(x + .5) / target.width, (y + .5) / target.height],
              gl_FragCoord: [x + .5, y + .5, .5, 1], gl_FragColor: [0, 0, 0, 0],
            }, context);

            if (!result) continue;
            const color = result.gl_FragColor as number[];
            if (!Array.isArray(color) || color.length !== 4 || !color.every(Number.isFinite))
              throw new Error(`CPU compute: ${pass.name} produced non-finite pixels at ${x},${y}.`);
            target.data.set(color, (y * target.width + x) * 4);
          }

          yield;
        }

        [state.front, state.back] = [state.back, state.front];
      }

      this.config.set(pass.name, config);
    }
  }

  outputs(): Record<string, SerializedTexture> {
    return Object.fromEntries(Object.entries(this.descriptor.outputs).map(([uniform, name]) => {
      const state = this.buffers.get(name);
      if (!state) throw new Error(`CPU compute: unknown output ${name}.`);

      return [uniform, state.front];
    }));
  }
}
