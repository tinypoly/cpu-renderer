import * as THREE from "three";
import { cpuRendererData, type CpuRendererLightData, type CpuRendererMaterialData, type CpuRendererSceneData } from "./extensions.js";
import { traverseRendered } from "./sceneSerialization.js";

function valueSignature(value: unknown): unknown {
  if (value instanceof THREE.Texture) {
    if (value.matrixAutoUpdate)
      value.updateMatrix();

    return [
      value.uuid,
      value.version,
      value.channel,
      value.mapping,
      value.minFilter, value.magFilter, value.generateMipmaps, value.anisotropy, value.colorSpace,
      value.wrapS, value.wrapT, value.flipY,
      value.mipmaps.map(level => {
        const mip = level as { width?: number; height?: number; image?: { width?: number; height?: number }[] };

        return [mip.width, mip.height, mip.image?.map(face => [face.width, face.height])];
      }),
      value.matrix.toArray(),
      value.image?.width,
      value.image?.height,
    ];
  }

  if (value instanceof THREE.Color || value instanceof THREE.Vector2 || value instanceof THREE.Vector3
    || value instanceof THREE.Vector4
    || value instanceof THREE.Matrix3
    || value instanceof THREE.Matrix4)
    return (value as THREE.Vector3).toArray();
  if (typeof value === "number" || typeof value === "string" || typeof value === "boolean" || value === null)
    return value;

  return undefined;
}

/** What the render reads from a material's renderer options; `time` changes every frame and is left out. */
function materialDataSignature(data: CpuRendererMaterialData): string {
  return JSON.stringify([Boolean(data.planarCapture), data.compute && [data.compute.id, data.compute.data]]);
}

export function sceneSignature(scene: THREE.Scene): string {
  scene.updateMatrixWorld(true);
  const fog = scene.fog as (THREE.Fog & THREE.FogExp2) | null;

  const parts: unknown[] = [
    cpuRendererData<CpuRendererSceneData>(scene).sky,
    fog && [fog.color.toArray(), fog.near, fog.far, fog.density],
  ];

  traverseRendered(scene, object => {
    const mesh = object as THREE.Mesh;

    if ((mesh.isMesh || (object as THREE.Points).isPoints
      || (object as THREE.Line).isLine || (object as THREE.Sprite).isSprite) && mesh.geometry) {
      const g = mesh.geometry, attributes = Object.entries(g.attributes).map(([name, a]) => [name, "version" in a ? a.version : a.data.version, a.count]);

      const materials = (Array.isArray(mesh.material) ? mesh.material : [mesh.material]).map(m => [
        m.uuid, m.version,
        Object.fromEntries(
          Object.entries(m).map(([k, v]) => [k, valueSignature(v)]).filter(([, v]) => v !== undefined),
        ),
        (m as THREE.ShaderMaterial).vertexShader, (m as THREE.ShaderMaterial).fragmentShader,
        Object.fromEntries(Object.entries((m as THREE.ShaderMaterial).uniforms ?? cpuRendererData<CpuRendererMaterialData>(m).uniforms ?? {}).filter(([name]) => name !== "time").map(([k, v]) => [k, valueSignature((v as THREE.IUniform).value)])),
        materialDataSignature(cpuRendererData<CpuRendererMaterialData>(m)),
      ]);

      const instance = mesh as THREE.InstancedMesh;
      parts.push([
        mesh.uuid,
        object.type,
        (object as THREE.Sprite).center?.toArray(),
        mesh.matrixWorld.toArray(),
        g.uuid,
        attributes,
        g.index?.version,
        g.drawRange,
        g.groups,
        materials,
        mesh.castShadow,
        mesh.receiveShadow,
        mesh.renderOrder,
        mesh.morphTargetInfluences,
        instance.count,
        instance.instanceMatrix?.version,
        instance.instanceColor?.version,
      ]);
    }

    const light = object as THREE.Light;
    const volumetric = cpuRendererData<CpuRendererLightData>(light).volumetric;
    if (light.isLight)
      parts.push([
        light.uuid, light.matrixWorld.toArray(), JSON.stringify(volumetric ?? null),
        Object.fromEntries(Object.entries(light).map(([k, v]) => [k, valueSignature(v)])),
        (light as THREE.DirectionalLight).target?.getWorldPosition(new THREE.Vector3()).toArray(),
      ]);
  });

  return JSON.stringify(parts);
}
