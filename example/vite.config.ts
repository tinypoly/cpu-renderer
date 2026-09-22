import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import { defineConfig, type Plugin } from "vite";

// Multiple render workers share memory through SharedArrayBuffer, which needs a cross-origin isolated page.
const isolationHeaders = {
  "Cross-Origin-Opener-Policy": "same-origin",
  "Cross-Origin-Embedder-Policy": "require-corp",
};

/**
 * GitHub Pages cannot send those headers, so the built page ships coi-serviceworker,
 * which adds them from a service worker. It must be a separate, unbundled file on the same origin.
 */
function crossOriginIsolationWorker(): Plugin {
  const fileName = "coi-serviceworker.min.js";

  return {
    name: "cross-origin-isolation-worker",
    apply: "build",
    generateBundle() {
      const source = createRequire(import.meta.url).resolve(`coi-serviceworker/${fileName}`);
      this.emitFile({ type: "asset", fileName, source: readFileSync(source, "utf8") });
    },
    transformIndexHtml: () => [{ tag: "script", attrs: { src: fileName }, injectTo: "head-prepend" }],
  };
}

export default defineConfig({
  root: fileURLToPath(new URL(".", import.meta.url)),
  // Relative paths so the build works under any GitHub Pages sub-path.
  base: "./",
  resolve: {
    // The example imports the package's build, exactly as an app installed from npm would.
    alias: { "@tinypoly/cpu-renderer": fileURLToPath(new URL("../dist/index.js", import.meta.url)) },
  },
  plugins: [crossOriginIsolationWorker()],
  server: { headers: isolationHeaders },
  preview: { headers: isolationHeaders },
  worker: { format: "es" },
});
