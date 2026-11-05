import { describe, expect, it } from "vitest";
import * as THREE from "three";
import { serializeScene } from "./sceneSerialization.js";

function solidTexture(value: number) {
  return new THREE.DataTexture(new Uint8Array([value, value, value, 255]), 1, 1);
}

describe("serializeScene progress", () => {
  it("reports the textures read out of the total, starting at zero", async() => {
    const scene = new THREE.Scene();
    for (const value of [0, 128])
      scene.add(new THREE.Mesh(new THREE.BoxGeometry(), new THREE.MeshBasicMaterial({ map: solidTexture(value) })));
    const steps: string[] = [];
    await serializeScene(scene, { onProgress: (completed, total) => steps.push(`${completed}/${total}`) });
    expect(steps).toEqual(["0/2", "1/2", "2/2"]);
  });

  it("reports an empty total for a scene without textures", async() => {
    const scene = new THREE.Scene();
    scene.add(new THREE.Mesh(new THREE.BoxGeometry(), new THREE.MeshBasicMaterial()));
    const steps: string[] = [];
    await serializeScene(scene, { onProgress: (completed, total) => steps.push(`${completed}/${total}`) });
    expect(steps).toEqual(["0/0"]);
  });
});
