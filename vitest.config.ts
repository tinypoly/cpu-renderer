import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    environment: "node",
    // The GI and memory cases render full images and exceed 5 s on a slow machine.
    testTimeout: 30000,
  },
});
