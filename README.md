# @tinypoly/cpu-renderer

A software renderer for [Three.js](https://threejs.org) scenes. It takes an ordinary `THREE.Scene` and renders it on the CPU across a pool of Web Workers, bucket by bucket, into a 2D canvas. It never creates a WebGL context.

It powers the render mode of the [Tinypoly](https://tinypoly.com) editor.

**[Live example](https://tinypoly.github.io/cpu-renderer/)**

- Physically based shading: Standard and Physical materials (clearcoat, sheen, transmission, anisotropy), Lambert, Phong, Toon and Basic.
- Perspective and orthographic cameras; meshes, lines (`Line`, `LineSegments`, `LineLoop`), dashed lines, sprites and native `PointsMaterial`.
- Your custom `ShaderMaterial` GLSL, run by a built-in interpreter that compiles to closures. It never uses `eval`.
- BVH ray-traced shadows with soft penumbrae, and ambient occlusion.
- Diffuse global illumination with an irradiance cache, an SVGF-style denoiser and environment importance sampling.
- HDR or gradient environments, depth of field with polygonal bokeh, fog, vignette and grain, and ACES, AgX or Neutral tone mapping.
- Progressive output: pixels appear as they finish on your canvas, and the render can be paused and resumed.
- With cross-origin isolation, the workers share the prepared scene through `SharedArrayBuffer`, so adding a worker costs almost no memory.

## Install

```sh
npm install @tinypoly/cpu-renderer three
```

`three` is a peer dependency (0.185 or newer).

## Usage

```ts
import * as THREE from "three";
import { CpuRenderer } from "@tinypoly/cpu-renderer";

const scene = new THREE.Scene();
const sphere = new THREE.Mesh(new THREE.SphereGeometry(1, 48, 24), new THREE.MeshPhysicalMaterial({ color: "tomato" }));
sphere.castShadow = sphere.receiveShadow = true;
scene.add(sphere);

const camera = new THREE.PerspectiveCamera(40, 1, 0.1, 100);
camera.position.set(0, 1, 5);
camera.lookAt(0, 0, 0);

const renderer = new CpuRenderer(document.querySelector("canvas")!, {
  scene,
  camera,
  environment: { kind: "gradient", topColor: "#9cb8d8", bottomColor: "#2b2a28" },
});

renderer.on("progress", ({ completed, total }) => console.log(`${completed} / ${total}`));
await renderer.render();
// The canvas holds the image: renderer.canvas.toBlob(...) saves it as a PNG.
```

The image size is the canvas' CSS size. Only meshes with `castShadow` / `receiveShadow` take part in shadows.

Setters record the inputs; `render()` starts a frame with the current ones:

```ts
camera.position.x += 1;                  // the camera is read at every render()
renderer.setSettings({ globalIllumination: { enabled: true, samples: 128 } });
await renderer.render();
```

## API

### `new CpuRenderer(canvas, options?)`

`canvas` is an `HTMLCanvasElement`; its 2D context receives the buckets. Every option has a setter of the same name.

| Option | Type | |
| --- | --- | --- |
| `scene` | `THREE.Scene` or the result of `serializeScene` | A Three scene is serialized in the background. Defaults to an empty scene. |
| `camera` | `THREE.PerspectiveCamera`, `THREE.OrthographicCamera` or `CameraState` | A Three camera is read at every `render()`. Defaults to a perspective camera at (5, 5, 5) looking at the origin. |
| `settings` | `RenderSettingsInput` | Merged over `DEFAULT_RENDER_SETTINGS`, group by group. See [Settings](#settings). |
| `environment` | `RenderEnvironment` | See [Environment](#environment). Defaults to `DEFAULT_RENDER_ENVIRONMENT`, a gradient. |
| `width`, `height` | `number` | Image size in CSS pixels. Without them, each frame measures the canvas' CSS size. |
| `pixelRatio` | `number` | Defaults to `window.devicePixelRatio`. |
| `workers` | `number` | Pool size. Defaults to `defaultWorkerCount()`: cores minus one, at most 8. Capped at `MAX_RENDER_WORKERS` (32). |
| `createWorker` | `() => Worker` | Worker factory. See [Bundlers and the worker](#bundlers-and-the-worker). |

### Methods

| Method | |
| --- | --- |
| `render()` | Starts a frame with the current inputs and discards the one in progress. Resolves when the canvas holds the finished image. Rejects with an `AbortError` (see `isAbortError`) when a later `render()` or `dispose()` replaces the frame, and with the worker's error when the frame fails. |
| `setScene(scene)` | A Three scene is serialized in the background; `render()` waits for it. |
| `setCamera(camera)` | A Three camera is read as it is at every `render()`, so moving it or changing its aspect only needs a new frame. A `CameraState` from `serializeCamera` stays fixed. |
| `setSettings(settings)` | Merges the given fields into the current settings, group by group: `{ fog: { enabled: true } }` keeps the other fog fields. |
| `setEnvironment(environment)` | Replaces the environment. A change of intensity or rotation alone does not load the image again. |
| `setSize(width, height, pixelRatio?)` | Fixes the image size in CSS pixels. |
| `pause()`, `resume()` | Stops or continues every worker on the same frame. |
| `dispose()` | Terminates the workers. |
| `settings` | The settings the next frame renders with. |
| `rendering`, `paused` | Whether a frame is in flight and whether the workers are paused. |
| `canvas` | The output canvas passed to the constructor. |
| `workerCount` | Number of workers actually running. |

### Events

`renderer.on(event, listener)` returns the function that removes the listener; `once` and `off` exist too.

| Event | Payload | |
| --- | --- | --- |
| `start` | `{ width, height }` | The workers prepared the frame and are shading. The canvas has the render size. |
| `progress` | `{ completed, total, phase }` | Work done out of `total`. `phase` is `"prepare"`, `"cache"` (irradiance cache), `"shade"` (buckets) or `"denoise"`. |
| `bucket` | `{ worker, bucket, width, height }` | The bucket a worker is on, or `null` when it is idle. Useful to draw progress frames. |
| `complete` | | The image on the canvas is finished. `render()` resolves right after. |
| `error` | `Error` | The frame failed. Unsupported shader syntax is reported here. `render()` rejects with the same error. |

### Settings

`RenderSettings`, grouped by feature. Defaults are `DEFAULT_RENDER_SETTINGS`; `mergeSettings(base, input)` applies a partial `RenderSettingsInput` the way `setSettings` does.

| Field | Default | |
| --- | --- | --- |
| `renderScale` | `1` | Multiplier on the image size. |
| `samples` | `4` | Antialiasing samples per pixel. |
| `tileSize` | `64` | Bucket size in pixels. |
| `toneMapping` | `"aces"` | `"aces"`, `"agx"`, `"neutral"` or `"linear"`. |
| `exposure` | `1` | |
| `background.mode` | `"environment"` | `"environment"`, `"color"` or `"transparent"`. |
| `background.color` | `"#1a1a1a"` | Used by `mode: "color"`. |
| `shadows.enabled` | `true` | Ray-traced shadows. |
| `shadows.softness` | `1` | Angular diameter of the lights in degrees. `0` gives hard shadows. |
| `ambientOcclusion.enabled` | `true` | |
| `ambientOcclusion.distance` | `1.5` | Ray length in world units. |
| `ambientOcclusion.intensity` | `1` | |
| `globalIllumination.enabled` | `false` | Diffuse path tracing. |
| `globalIllumination.samples` | `64` | Paths per shaded pixel. |
| `globalIllumination.bounces` | `3` | |
| `globalIllumination.intensity` | `1` | |
| `globalIllumination.clamp` | `0` | Maximum indirect sample luminance. `0` disables clamping. |
| `globalIllumination.cache` | `true` | Compute indirect light on a pixel grid and interpolate. |
| `globalIllumination.cacheSpacing` | `4` | Grid spacing in pixels. |
| `globalIllumination.denoise` | `true` | Denoise the indirect light after the last bucket. |
| `depthOfField.enabled` | `false` | |
| `depthOfField.focusDistance` | `10` | World units. |
| `depthOfField.aperture` | `2.8` | f-number. |
| `depthOfField.blades` | `0` | Polygon sides of the bokeh. `0` is circular. |
| `fog.enabled` | `false` | Linear fog over every surface. Without it, `scene.fog` applies. |
| `fog.color` | `"#b8c4d0"` | |
| `fog.near`, `fog.far` | `10`, `60` | View depth. |
| `vignette` | `0` | `0` to `1`. |
| `grain` | `0` | `0` to `1`. |

### Environment

`RenderEnvironment`. Lights the scene and paints the background when `background.mode` is `"environment"`.

```ts
{ kind: "hdri", url: "studio.hdr" }                                            // equirectangular .hdr
{ kind: "gradient", topColor: "#9cb8d8", bottomColor: "#2b2a28", exponent: 2 } // exponent defaults to 2
{ kind: "hdri", url: "studio.hdr", intensity: 0.8, rotation: 90 }              // intensity 1 and rotation 0 by default
```

### Object options

The renderer reads optional settings from `userData.cpuRenderer` on Three objects. A plain scene needs none of them.

| Where | Field | |
| --- | --- | --- |
| Any object | `exclude` | Leaves the object and its children out of the render. |
| Light | `volumetric` | Scatters the light in the air: `true`, or a medium `{ density, anisotropy, spread }`. |
| Scene | `sky` | A `ProceduralSky`: a shader that replaces the environment. |
| Material | `time` | Value of the shader's `time` uniform. |
| Material | `planarCapture` | Renders planar reflection and refraction textures for a flat surface. See `CpuRendererMaterialData`. |
| Material | `compute` | A `MaterialCompute`: fragment passes baked on the CPU into the material's uniforms. |
| Material | `uniforms` | The live uniforms of a material patched in `onBeforeCompile`. |
| Material | `prepare` | Called with the scene before it is serialized. |

```ts
light.userData.cpuRenderer = { volumetric: { density: 0.05 } } satisfies CpuRendererLightData;
```

### Helpers

| Function | |
| --- | --- |
| `serializeScene(scene)` | Async. The snapshot `setScene` builds from a Three scene, to reuse across renderers or build off the main thread. |
| `serializeCamera(camera)` | A fixed `CameraState` snapshot of a Three camera, including zoom and view offsets. |
| `sceneSignature(scene)` | A string that changes whenever something the renderer reads changes. Compare it before re-serializing an interactive scene. |
| `isAbortError(error)` | Whether a `render()` rejection means the frame was replaced rather than failed. |
| `shareable()` | Whether the page is cross-origin isolated, and so whether more than one worker can run. |
| `defaultWorkerCount()` | The pool size used when `workers` is omitted. |

### Engine

`@tinypoly/cpu-renderer/engine` exports the engine behind `CpuRenderer`: `CpuRasterizer`, `CpuScenePipeline`, `CpuShader`, `CpuComputePipeline`, the worker protocol and the shared memory helpers, for custom workers, Node scripts and tests. It takes the flat `FrameSettings`, built from `RenderSettings` by `toFrameSettings`. It is lower level than the main entry and changes more often. [ARCHITECTURE.md](ARCHITECTURE.md) describes it.

## Bundlers and the worker

The client starts its workers with `new Worker(new URL("./worker.js", import.meta.url), { type: "module" })`, which Vite, webpack 5, Rspack and Parcel bundle.

With **Vite**, exclude the package from dependency pre-bundling, or the dev server moves the client away from `worker.js`:

```ts
// vite.config.ts
export default defineConfig({
  optimizeDeps: { exclude: ["@tinypoly/cpu-renderer"] },
  worker: { format: "es" },
});
```

If your setup cannot follow the pattern, pass your own factory. The worker entry is exported as `@tinypoly/cpu-renderer/worker`:

```ts
import RenderWorker from "@tinypoly/cpu-renderer/worker?worker"; // Vite syntax

new CpuRenderer(canvas, { createWorker: () => new RenderWorker() });
```

## Cross-origin isolation

More than one worker needs `SharedArrayBuffer`, which browsers only allow on cross-origin isolated pages:

```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

Without these headers the renderer still works, with a single worker. On hosts where you cannot set headers, such as GitHub Pages, [coi-serviceworker](https://github.com/gzuidhof/coi-serviceworker) adds them from a service worker.

## Limits

This is a functional CPU renderer. It does not conform to all of Three.js or WebGL. GPU block-compressed textures are only read when transcoded to RGBA. Unsupported shader syntax is reported as an error instead of silently substituting another material. [ARCHITECTURE.md](ARCHITECTURE.md) has the full compatibility table.

## License

[MIT](LICENSE)
