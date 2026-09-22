import * as THREE from "three";
import { EffectComposer } from "three/examples/jsm/postprocessing/EffectComposer.js";
import { OutputPass } from "three/examples/jsm/postprocessing/OutputPass.js";
import { RenderPass } from "three/examples/jsm/postprocessing/RenderPass.js";
import type { RenderToneMapping } from "@tinypoly/cpu-renderer";
import { studioEnvironment } from "./studio";

const TONE_MAPPING: Record<RenderToneMapping, THREE.ToneMapping> = {
  aces: THREE.ACESFilmicToneMapping,
  agx: THREE.AgXToneMapping,
  neutral: THREE.NeutralToneMapping,
  linear: THREE.LinearToneMapping,
};

/**
 * A WebGL view of the same Three scene, shown while the camera moves and underneath the CPU image as it fills in.
 * It renders like the CPU renderer does: materials, fog and blending in linear HDR light, then one output pass that
 * tone maps and encodes. Drawing straight to the canvas would tone map each material and fog the result afterwards.
 */
export class ScenePreview {
  private renderer: THREE.WebGLRenderer;
  private composer: EffectComposer;
  private pass: RenderPass;
  private sky: THREE.Texture;
  private scene: THREE.Scene | null = null;

  /** The studio as a .hdr file, for the CPU renderer: the preview lights with the same pixels. */
  readonly environmentUrl: string;

  constructor(canvas: HTMLCanvasElement) {
    this.renderer = new THREE.WebGLRenderer({ canvas });
    this.renderer.shadowMap.enabled = true;

    this.composer = new EffectComposer(this.renderer,
      new THREE.WebGLRenderTarget(1, 1, { type: THREE.HalfFloatType, samples: 4 }));

    this.pass = new RenderPass(new THREE.Scene(), new THREE.Camera());
    this.composer.addPass(this.pass);
    this.composer.addPass(new OutputPass());
    const studio = studioEnvironment();
    this.sky = studio.texture;
    this.environmentUrl = studio.url;
  }

  /** Call after serializing: the CPU renderer does not read `background` or `environment`, but keeps them out anyway. */
  setScene(scene: THREE.Scene, enclosed = false) {
    // A closed room sees none of the studio: the viewer turns the CPU environment off too, background included.
    scene.background = enclosed ? new THREE.Color(0) : this.sky;
    scene.environment = enclosed ? null : this.sky;
    this.scene = scene;
  }

  /** Blanks the canvas and forgets the scene, so nothing redraws the previous scene before the next one is set. */
  clear() {
    this.scene = null;
    this.renderer.setRenderTarget(null);
    this.renderer.clear();
  }

  setToneMapping(toneMapping: RenderToneMapping, exposure: number) {
    this.renderer.toneMapping = TONE_MAPPING[toneMapping];
    this.renderer.toneMappingExposure = exposure;
  }

  resize(width: number, height: number) {
    this.renderer.setPixelRatio(window.devicePixelRatio);
    this.renderer.setSize(Math.max(1, width), Math.max(1, height), false);
    this.composer.setPixelRatio(window.devicePixelRatio);
    this.composer.setSize(Math.max(1, width), Math.max(1, height));
  }

  render(camera: THREE.Camera) {
    if (!this.scene)
      return;
    this.pass.scene = this.scene;
    this.pass.camera = camera;
    this.composer.render();
  }
}
