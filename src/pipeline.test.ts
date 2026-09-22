import { describe, expect, it } from "vitest";
import * as THREE from "three";
import { CpuScenePipeline } from "./pipeline.js";
import { CpuEnvironment } from "./environment.js";
import { serializeCamera } from "./camera.js";
import { serializeScene, type SerializedScene, type SerializedTexture } from "./sceneSerialization.js";
import { DEFAULT_FRAME_SETTINGS, type FrameSettings } from "./frameSettings.js";
import type { GpgpuPipelineDescriptor } from "./shaderTypes.js";
import type { CpuRendererMaterialData } from "./extensions.js";

const settings: FrameSettings = {
  ...DEFAULT_FRAME_SETTINGS, maxSamples: 1, tileSize: 8, ambientOcclusion: false, shadows: false,
};

function drain<T>(job: Generator<void, T>): T {
  let step = job.next();
  while (!step.done)
    step = job.next();

  return step.value;
}

const environment = () => CpuEnvironment.load({ kind: "gradient", topColor: "#ffffff", bottomColor: "#404040" });

function camera(position: [number, number, number]) {
  const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 50);
  camera.position.set(...position);
  camera.lookAt(0, 0, 0);

  return serializeCamera(camera);
}

/** A 2 × 2 buffer that each run increments by one: its values count how many times the simulation ran. */
const simulation: GpgpuPipelineDescriptor = {
  buffers: [{ name: "height", width: 2, height: 2, pingPong: true }],
  passes: [{
    name: "grow", inputs: ["height"], output: "height", trigger: "everyFrame",
    fragmentShader: "void main() { gl_FragColor = vec4(texture2D(height, vUv).x + 1.0, 0.0, 0.0, 1.0); }",
  }],
  outputs: { uHeight: "height" },
};

/** A flat surface with the renderer's material options; its uuid names its captures. */
function shaderMesh(data: CpuRendererMaterialData, uuid = "water") {
  const material = new THREE.ShaderMaterial({
    side: THREE.DoubleSide,
    uniforms: { uHeight: { value: null }, time: { value: 0 } },
    vertexShader: "void main() { gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }",
    fragmentShader: "void main() { gl_FragColor = vec4(1.0); }",
  });

  Object.assign(material, { uuid });
  material.userData.cpuRenderer = { time: 2.5, ...data } satisfies CpuRendererMaterialData;

  return new THREE.Mesh(new THREE.PlaneGeometry(4, 4).rotateX(-Math.PI / 2), material);
}

async function serialize(...objects: THREE.Object3D[]): Promise<SerializedScene> {
  const scene = new THREE.Scene();
  scene.add(...objects, new THREE.DirectionalLight("#ffffff", 2));

  return (await serializeScene(scene)).scene;
}

const texture = (scene: SerializedScene, id: string) => scene.textures.find(t => t.id === id) as SerializedTexture;

describe("CpuScenePipeline", () => {
  it("bakes compute outputs into textures bound to the material, without touching the source scene", async() => {
    const mesh = shaderMesh({ compute: { pipeline: simulation, id: "water" } });
    // Two meshes share the material: the simulation runs once and its texture is listed once.
    const source = await serialize(mesh, mesh.clone());
    const pipeline = new CpuScenePipeline(source);
    const scene = drain(pipeline.prepare(camera([0, 3, 5]), settings, 16, 16, await environment()));
    const [material] = scene.meshes[0].materials;
    expect(material.shader!.uniforms.uHeight).toEqual({ texture: "water:height" });
    expect(material.shader!.uniforms.time).toBe(2.5);
    expect(scene.meshes[1].materials[0]).toBe(material);
    expect(scene.textures.filter(t => t.id === "water:height")).toHaveLength(1);
    expect(Array.from(texture(scene, "water:height").data).filter((_, i) => i % 4 === 0)).toEqual([1, 1, 1, 1]);
    expect(source.textures).toHaveLength(0);
    expect(source.meshes[0].materials[0].shader!.uniforms.uHeight).toBeNull();
  });

  it("keeps a completed bake across frames and never caches an interrupted one", async() => {
    const pipeline = new CpuScenePipeline(await serialize(shaderMesh({ compute: { pipeline: simulation, id: "water" } })));
    const env = await environment();
    // A frame cancelled in the middle of the simulation: the generator is dropped after one step.
    pipeline.prepare(camera([0, 3, 5]), settings, 16, 16, env).next();
    const first = drain(pipeline.prepare(camera([0, 3, 5]), settings, 16, 16, env));
    const second = drain(pipeline.prepare(camera([4, 2, 1]), { ...settings, maxSamples: 4 }, 32, 16, env));
    // Had the bake run again, the counter would read 2.
    expect(texture(second, "water:height")).toBe(texture(first, "water:height"));
    expect(texture(second, "water:height").data[0]).toBe(1);
  });

  it("restores a paused simulation from its snapshot instead of running it", async() => {
    const values = Float32Array.from({ length: 16 }, (_, i) => i);

    const pipeline = new CpuScenePipeline(await serialize(shaderMesh({ compute: { pipeline: simulation, id: "water",
      snapshot: { outputs: { uHeight: { width: 2, height: 2, data: values } } } } })));

    const scene = drain(pipeline.prepare(camera([0, 3, 5]), settings, 16, 16, await environment()));
    expect(Array.from(texture(scene, "water:height").data)).toEqual(Array.from(values));
  });

  it("captures planar reflection, environment reflection and refraction with depth for the material", async() => {
    const box = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), new THREE.MeshBasicMaterial({ color: "#ff0000" }));
    box.position.y = 1;
    const water = shaderMesh({ planarCapture: true });
    const pipeline = new CpuScenePipeline(await serialize(water, box));
    const view = camera([0, 3, 5]);
    const scene = drain(pipeline.prepare(view, settings, 16, 12, await environment()));
    const uniforms = scene.meshes[0].materials[0].shader!.uniforms;

    for (const name of ["tReflectionMap", "tEnvironmentReflectionMap", "tRefractionMap", "tRefractionDepth"]) {
      const capture = texture(scene, `capture:water:${name}`);
      expect(uniforms[name]).toEqual({ texture: capture.id });
      expect([capture.width, capture.height, capture.flipY]).toEqual([16, 12, true]);
    }

    expect(uniforms.reflectionMatrix).toHaveLength(16);
    expect(uniforms.refractionMatrix).toHaveLength(16);
    expect([uniforms.cameraNear, uniforms.cameraFar]).toEqual([view.near, view.far]);

    const alpha = (name: string) => Array.from(texture(scene, `capture:water:${name}`).data).filter((_, i) => i % 4 === 3);
    // The scene reflection has a transparent background: only the mirrored box covers it.
    expect(alpha("tReflectionMap").some(value => value === 0)).toBe(true);
    expect(alpha("tReflectionMap").some(value => value > 0)).toBe(true);
    // The environment reflection holds no geometry and covers every pixel.
    expect(alpha("tEnvironmentReflectionMap").every(value => value > 0)).toBe(true);
    const depth = Array.from(texture(scene, "capture:water:tRefractionDepth").data).filter((_, i) => i % 4 === 0);
    expect(depth.every(value => value >= 0 && value <= 1)).toBe(true);
  });

  it("leaves every planar surface out of the captures, so they never depend on each other", async() => {
    const lower = shaderMesh({ planarCapture: true });
    const upper = shaderMesh({ planarCapture: true }, "pond");
    upper.position.y = 0.5;
    const pipeline = new CpuScenePipeline(await serialize(lower, upper));
    const scene = drain(pipeline.prepare(camera([0, 3, 5]), settings, 16, 12, await environment()));
    // The upper surface is above the lower one's mirror plane: included, it would cover its reflection.
    const reflection = texture(scene, "capture:water:tReflectionMap");
    expect(Array.from(reflection.data).filter((_, i) => i % 4 === 3).every(value => value === 0)).toBe(true);
  });

  it("caps planar captures at 1024 pixels on the long side", async() => {
    const pipeline = new CpuScenePipeline(await serialize(shaderMesh({ planarCapture: true })));
    const scene = drain(pipeline.prepare(camera([0, 3, 5]), settings, 2048, 4, await environment()));
    const capture = texture(scene, "capture:water:tRefractionMap");
    expect([capture.width, capture.height]).toEqual([1024, 2]);
  });
});
