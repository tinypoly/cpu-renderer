import { describe, expect, it, vi } from "vitest";
import * as THREE from "three";
import { serializeScene } from "./sceneSerialization.js";
import type { CpuRendererMaterialData, CpuRendererSceneData, ProceduralSky } from "./extensions.js";
import type { GpgpuPipelineDescriptor } from "./shaderTypes.js";

const pipeline: GpgpuPipelineDescriptor = {
  buffers: [{ name: "field", width: 1, height: 1 }],
  passes: [{ name: "fill", inputs: [], output: "field", trigger: "everyFrame",
    fragmentShader: "void main() { gl_FragColor = vec4(1.0); }" }],
  outputs: { uField: "field" },
};

function shaderMaterial(data: CpuRendererMaterialData) {
  const material = new THREE.ShaderMaterial({ uniforms: { uField: { value: null }, uTint: { value: 1 } } });
  material.userData.cpuRenderer = data;

  return material;
}

async function serialize(...objects: THREE.Object3D[]) {
  const scene = new THREE.Scene();
  scene.add(...objects);

  return (await serializeScene(scene)).scene;
}

describe("userData.cpuRenderer", () => {
  it("leaves an excluded object and its children out of the render", async() => {
    const helper = new THREE.Mesh(new THREE.BoxGeometry(), new THREE.MeshBasicMaterial());
    helper.userData.cpuRenderer = { exclude: true };
    helper.add(new THREE.Mesh(new THREE.BoxGeometry(), new THREE.MeshBasicMaterial()));
    const kept = new THREE.Mesh(new THREE.BoxGeometry(), new THREE.MeshBasicMaterial());
    expect((await serialize(helper, kept)).meshes).toHaveLength(1);
  });

  it("carries a material's time, planar capture and simulation into the snapshot", async() => {
    const material = shaderMaterial({ time: 3, planarCapture: true, compute: { pipeline, id: "sim", data: { a: 1 } } });
    const [mesh] = (await serialize(new THREE.Mesh(new THREE.PlaneGeometry(), material))).meshes;
    expect(mesh.materials[0]).toMatchObject({
      time: 3, planarCapture: { id: material.uuid }, compute: { pipeline, id: "sim", data: { a: 1 } },
    });
    // Uniforms the render generates never carry the live GPU value.
    expect(mesh.materials[0].shader!.uniforms.uField).toBeNull();
    expect(mesh.materials[0].shader!.uniforms.uTint).toBe(1);
  });

  it("rejects a simulation without a pipeline or an id", async() => {
    const material = shaderMaterial({ compute: { pipeline, id: "" } });
    await expect(serialize(new THREE.Mesh(new THREE.PlaneGeometry(), material))).rejects.toThrow("without a pipeline or an id");
  });

  it("calls each material's prepare hook with the scene before serializing it", async() => {
    const prepare = vi.fn((scene: THREE.Scene) => expect(scene.isScene).toBe(true));
    const material = shaderMaterial({ prepare });
    await serialize(new THREE.Mesh(new THREE.PlaneGeometry(), material));
    expect(prepare).toHaveBeenCalledOnce();
  });

  it("reads the live uniforms of a material patched in onBeforeCompile", async() => {
    const material = new THREE.MeshStandardMaterial();
    const uniforms = { uWave: { value: 0.25 } };
    material.onBeforeCompile = shader => Object.assign(shader.uniforms, uniforms);
    material.userData.cpuRenderer = { uniforms, planarCapture: true } satisfies CpuRendererMaterialData;
    const [mesh] = (await serialize(new THREE.Mesh(new THREE.PlaneGeometry(), material))).meshes;
    expect(mesh.materials[0].planarCapture).toEqual({ id: material.uuid });
  });

  it("serializes the scene's procedural sky", async() => {
    const sky: ProceduralSky = { fragmentShader: "void main() {}", uniforms: { uSun: [0, 1, 0] }, intensity: 2,
      background: true };

    const scene = new THREE.Scene();
    scene.userData.cpuRenderer = { sky } satisfies CpuRendererSceneData;
    expect((await serializeScene(scene)).scene.sky).toEqual(sky);
    expect((await serializeScene(new THREE.Scene())).scene).not.toHaveProperty("sky");
  });
});
