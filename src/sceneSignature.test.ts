import { describe, expect, it } from "vitest";
import * as THREE from "three";
import { sceneSignature } from "./sceneSignature.js";

/** Asserts that `change` alters the signature, or leaves it as it was with `changes = false`. */
function expectChange(scene: THREE.Scene, change: () => void, changes = true) {
  const before = sceneSignature(scene);
  change();
  if (changes)
    expect(sceneSignature(scene)).not.toBe(before);
  else
    expect(sceneSignature(scene)).toBe(before);
}

function meshScene() {
  const scene = new THREE.Scene();
  const mesh = new THREE.Mesh(new THREE.BoxGeometry(), new THREE.MeshStandardMaterial());
  scene.add(mesh);

  return { scene, mesh };
}

describe("sceneSignature", () => {
  it("is stable while nothing changes", () => {
    const { scene } = meshScene();
    expect(sceneSignature(scene)).toBe(sceneSignature(scene));
  });

  it("follows transforms, material values, geometry versions and shadow flags", () => {
    const { scene, mesh } = meshScene();
    const material = mesh.material;
    expectChange(scene, () => mesh.position.x = 1);
    expectChange(scene, () => material.color.set("#ff0000"));
    expectChange(scene, () => material.roughness = 0.2);
    expectChange(scene, () => mesh.geometry.attributes.position.needsUpdate = true);
    expectChange(scene, () => mesh.castShadow = true);
    expectChange(scene, () => mesh.geometry.setDrawRange(0, 3));
  });

  it("follows interleaved attributes through their shared buffer", () => {
    const scene = new THREE.Scene(), geometry = new THREE.BufferGeometry();
    const buffer = new THREE.InterleavedBuffer(new Float32Array(9 * 2), 6);
    geometry.setAttribute("position", new THREE.InterleavedBufferAttribute(buffer, 3, 0));
    scene.add(new THREE.Mesh(geometry, new THREE.MeshBasicMaterial()));
    expectChange(scene, () => buffer.needsUpdate = true);
  });

  it("follows texture content, sampling and mip chains", () => {
    const { scene, mesh } = meshScene();
    const texture = new THREE.DataTexture(new Uint8Array(16), 2, 2);
    mesh.material.map = texture;
    expectChange(scene, () => texture.needsUpdate = true);
    expectChange(scene, () => texture.wrapS = THREE.RepeatWrapping);
    expectChange(scene, () => texture.offset.set(0.5, 0));
    expectChange(scene, () => texture.mipmaps = [{ data: new Uint8Array(4), width: 1, height: 1 }]);
  });

  it("follows shader uniforms but not the time uniform, which the renderer sets itself", () => {
    const scene = new THREE.Scene();

    const material = new THREE.ShaderMaterial({
      uniforms: { time: { value: 0 }, uTint: { value: new THREE.Color() } },
    });

    scene.add(new THREE.Mesh(new THREE.PlaneGeometry(), material));
    expectChange(scene, () => material.uniforms.time.value = 5, false);
    expectChange(scene, () => material.uniforms.uTint.value.set("#00ff00"));
  });

  it("follows lights, their targets and scene fog", () => {
    const scene = new THREE.Scene(), light = new THREE.DirectionalLight();
    scene.add(light, light.target);
    expectChange(scene, () => light.intensity = 4);
    expectChange(scene, () => light.target.position.set(1, 0, 0));
    expectChange(scene, () => scene.fog = new THREE.Fog("#ffffff", 1, 10));
    expectChange(scene, () => (scene.fog as THREE.Fog).far = 20);
  });

  it("ignores hidden objects and objects excluded from the render", () => {
    const { scene, mesh } = meshScene();
    const dome = new THREE.Mesh(new THREE.SphereGeometry(), new THREE.MeshBasicMaterial());
    dome.userData.cpuRenderer = { exclude: true };
    expectChange(scene, () => scene.add(dome), false);
    expectChange(scene, () => dome.position.y = 3, false);
    mesh.visible = false;
    expectChange(scene, () => mesh.material.color.set("#0000ff"), false);
  });

  it("follows instance matrices and colors", () => {
    const scene = new THREE.Scene();
    const instances = new THREE.InstancedMesh(new THREE.BoxGeometry(), new THREE.MeshBasicMaterial(), 2);
    scene.add(instances);
    expectChange(scene, () => instances.instanceMatrix.needsUpdate = true);
    expectChange(scene, () => instances.setColorAt(0, new THREE.Color("#ff0000")));
    expectChange(scene, () => instances.count = 1);
  });
});
