import { Matrix4, OrthographicCamera, PerspectiveCamera } from "three";
import type { CameraState } from "./settings.js";

/** Snapshot the actual projection, including zoom, film offset and view offsets. */
export function serializeCamera(camera: PerspectiveCamera | OrthographicCamera): CameraState {
  camera.updateWorldMatrix(true, false);
  camera.updateProjectionMatrix();

  const common = { matrixWorld: camera.matrixWorld.toArray(), near: camera.near, far: camera.far,
    projectionMatrix: camera.projectionMatrix.toArray(), zoom: camera.zoom };

  if (camera instanceof OrthographicCamera)
    return { ...common, type: "orthographic", left: camera.left, right: camera.right, top: camera.top, bottom: camera.bottom };

  return { ...common, type: "perspective", fov: camera.fov, aspect: camera.aspect };
}

export function cameraProjection(camera: CameraState, aspect: number): Matrix4 {
  if (camera.projectionMatrix) return new Matrix4().fromArray(camera.projectionMatrix);

  const projection = camera.type === "orthographic"
    ? new OrthographicCamera(camera.left, camera.right, camera.top, camera.bottom, camera.near, camera.far)
    : new PerspectiveCamera(camera.fov ?? 50, camera.aspect ?? aspect, camera.near, camera.far);

  projection.zoom = camera.zoom ?? 1;
  projection.updateProjectionMatrix();

  return projection.projectionMatrix;
}
