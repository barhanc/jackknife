# Jackknife in the browser

An experimental TypeScript port of the Python game. TypeGPU draws the yard and
truck and ruNNtime runs the existing ControlTransformer locally on WebGPU.
The HUD directly ports the desktop layout using its bitmap fonts, with accessible
DOM buttons over the drawing. Cached Canvas layers reproduce the desktop terrain,
lighting, road masks and snowfall. Training and the desktop game remain in the repository root.

Use Node 22.18+ and [uv](https://docs.astral.sh/uv/getting-started/installation/).
From the repository root:

```sh
uv sync
cd web
npm ci
npm run export-model
npm run dev
```

Open the localhost URL in a WebGPU-capable browser. A secure context (HTTPS or
localhost) is required. The first load downloads about 25 MiB of weights and
compiles the model's shaders. No backend, Python process, or inference service is
needed once the static site is built.

```sh
npm test
npm run build
npx playwright install chromium
npm run test:browser
```

`export-model` must run before tests/build. It generates weights, a starting
route, and Python reference cases under `public/models/`, plus shared bitmap
fonts, seeded snow, light sprites and desktop reference screenshots under
`public/assets/`. These generated files are ignored by Git. To export another checkpoint, run
`uv run python src/export_web_model.py --checkpoint models/your-checkpoint.pt`
from the repository root. Deploy the contents of `web/dist/` to a static host.
The build uses relative URLs and supports hosting under a subdirectory.

## Controls

W/S or arrows drive forward/reverse, A/D steer, Space pauses, M changes steering
mode, T changes simulation speed, E starts/cancels route drawing, R resets,
V toggles attention rays, and H toggles telemetry. The buttons expose the same
controls. Draw routes with a mouse, pen or touch; a straight lead-in is added.
Switching to autonomous mode selects 16× speed. Losing focus pauses the game.

The simulation uses the desktop's 50 Hz fixed timestep and a policy update every
four steps at all speed settings. Rendering runs independently. Policy calls are
serialized: every update uses the result for its own state, and resets/mode
changes discard stale results. Slow hardware can run below the requested speed;
the policy cadence is never reduced to fake 16× throughput.

## Layout and verification

- `src/simulation`: RK4 kinematics, route smoothing/features, 21 procedural route
  families, lot fitting and game state.
- `src/rendering`: TypeGPU truck geometry and cached desktop scenery/effect layers.
- `src/inference`: model architecture and weight loading through ruNNtime.
- `src/ui`: direct port of the Python HUD, controls and telemetry.
- `packages/`: local npm tarball of the pinned ruNNtime core, with types and MIT
  license. The upstream npm package does not expose this API; see packages/README.md.
- `scripts/pack-runntime.mjs`: rebuilds that tarball from the pinned upstream source.
- `../src/export_web_model.py`: checkpoint export and reference data generation.
- `../src/export_web_assets.py`: shared desktop assets and screenshot references.

Unit tests compare physics, route smoothing and policy tokens against Python,
and cover controller cadence and stale asynchronous results. The browser test
checks real GPU logits/attention against PyTorch, drives to a manual jackknife,
draws a route, and completes the default route autonomously at 16×. Screenshot
tests compare five matching desktop/browser scenes across three viewport sizes.

New Route generates from the same 21 families as the desktop game, with the same
parameter ranges, smoothing, rotations/reflections, yard fitting and retry rules.
It generates fresh routes locally; no route bank or Python server is used.
Recorded random draws verify each family against Python. Floating-point rounding
at an exact station-spacing boundary can produce one extra station across runtimes.

The desktop is the visual reference. The truck snaps to integer screen pixels,
with a minimum one-pixel detail stroke; geometry edges can still differ slightly
between Pygame, Canvas and WebGPU rasterisation. Touch route drawing works;
driving currently requires a keyboard. Inference telemetry uses a wall-time
exponential moving average with a half-second time constant at every speed.
