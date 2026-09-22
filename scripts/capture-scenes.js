// Renders every example scene on the CPU in headless Chrome and saves the gallery pictures to
// example/assets/scenes/<id>.webp. Start the example first (`pnpm dev`), then run:
//
//   node scripts/capture-scenes.js [url] [scene ids...]
//
// Chrome is taken from CHROME_PATH, or the usual macOS location.
import { spawn } from "node:child_process";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const [url = "http://localhost:5173/", ...only] = process.argv.slice(2);
const chromePath = process.env.CHROME_PATH ?? "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const output = new URL("../example/assets/scenes/", import.meta.url);
const port = 9335;
// The viewer at 1300 × 674 CSS pixels and 2× density: a 2008 × 1348 render, saved 960 pixels wide.
const viewport = { width: 1300, height: 674, deviceScaleFactor: 2, mobile: false };
const width = 960;
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));

const profile = mkdtempSync(join(tmpdir(), "capture-scenes-"));

const chrome = spawn(chromePath, [
  "--headless=new", `--remote-debugging-port=${port}`, `--user-data-dir=${profile}`, "about:blank",
], { stdio: "ignore" });

try {
  let target;

  for (let attempt = 0; attempt < 50 && !target; attempt++) {
    await sleep(200);

    try {
      const targets = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
      target = targets.find(t => t.type === "page");
    } catch {
      // Chrome is still starting.
    }
  }

  if (!target)
    throw new Error(`Chrome did not start (${chromePath}). Set CHROME_PATH.`);
  const socket = new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((resolve, reject) => { socket.onopen = resolve; socket.onerror = reject; });
  const pending = new Map();
  let nextId = 0;

  socket.onmessage = event => {
    const message = JSON.parse(event.data);
    pending.get(message.id)?.(message);
    pending.delete(message.id);
  };

  const send = (method, params = {}) => new Promise(resolve => {
    pending.set(++nextId, resolve);
    socket.send(JSON.stringify({ id: nextId, method, params }));
  });

  const evaluate = async expression => {
    const { result } = await send("Runtime.evaluate", { expression, awaitPromise: true, returnByValue: true });
    if (result.exceptionDetails)
      throw new Error(result.exceptionDetails.exception?.description ?? "evaluation failed");

    return result.result.value;
  };

  const open = async address => {
    await send("Page.navigate", { url: "about:blank" });
    await send("Page.navigate", { url: address });

    for (let attempt = 0; attempt < 100; attempt++) {
      await sleep(100);
      if (await evaluate("document.readyState") === "complete")
        return;
    }

    throw new Error(`${address} did not load. Is the example running (pnpm dev)?`);
  };

  await send("Page.enable");
  await send("Emulation.setDeviceMetricsOverride", viewport);
  await open(url);
  const ids = await evaluate("[...document.querySelectorAll('.scene-card')].map(card => card.hash.slice(1))");
  mkdirSync(output, { recursive: true });

  for (const id of only.length ? only : ids) {
    const started = performance.now();
    await open(`${url}#${id}`);
    let state = "";

    // Eight minutes is ample for every scene at this size, depth of field at 16 samples included.
    for (let attempt = 0; attempt < 4800 && state !== "done"; attempt++) {
      await sleep(100);
      state = await evaluate("document.getElementById('phase')?.dataset.state ?? ''");
      if (state === "error")
        throw new Error(`${id}: ${await evaluate("document.getElementById('status').textContent")}`);
    }

    if (state !== "done")
      throw new Error(`${id}: the render did not finish.`);

    const picture = await evaluate(`(() => {
      const source = document.getElementById("output"), target = document.createElement("canvas");
      target.width = ${width};
      target.height = Math.round(${width} * source.height / source.width);
      const context = target.getContext("2d");
      context.imageSmoothingQuality = "high";
      context.drawImage(source, 0, 0, target.width, target.height);
      return target.toDataURL("image/webp", 0.86);
    })()`);

    const file = new URL(`${id}.webp`, output);
    writeFileSync(file, Buffer.from(picture.split(",")[1], "base64"));
    console.log(`${id}: ${((performance.now() - started) / 1000).toFixed(1)} s → ${file.pathname}`);
  }

  socket.close();
} finally {
  // Chrome keeps writing to its profile until it exits.
  const exited = new Promise(resolve => chrome.once("exit", resolve));
  chrome.kill();
  await exited;
  rmSync(profile, { recursive: true, force: true });
}
