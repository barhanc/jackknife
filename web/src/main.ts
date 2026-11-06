/** Browser entry point: rendering and input remain live during asynchronous inference. */
import tgpu from 'typegpu';
import { createWorkerPolicy } from './inference/workerPolicy';
import { Game } from './simulation/game';
import { sampleRoute } from './simulation/generator';
import { drawnRoute, tokensFor, type Route } from './simulation/path';
import { axles, type Point } from './simulation/physics';
import { createRenderer } from './rendering/renderer';
import { createHud } from './ui/hud';
import { loadAssets } from './rendering/assets';
import './ui/style.css';

const app = document.querySelector<HTMLDivElement>('#app')!;
const hud = createHud(app);
async function boot() {
  if (!navigator.gpu)
    throw new Error(
      'This game needs a WebGPU-capable browser. Try an up-to-date Chrome, Edge or Safari on a supported device.',
    );
  const root = await tgpu.init({ device: { optionalFeatures: ['subgroups', 'shader-f16'] } });
  root.device.addEventListener('uncapturederror', (e) => console.error('WebGPU:', e.error.message));
  hud.loading('Loading the steering network…');
  const [policy, routeResponse, assets] = await Promise.all([
    createWorkerPolicy(),
    fetch(`${import.meta.env.BASE_URL}models/track.json`),
    loadAssets(),
  ]);
  if (!routeResponse.ok) throw new Error('Starting route missing. Run npm run export-model.');
  const route: Route = await routeResponse.json();
  const game = new Game(route, policy);
  if (new URLSearchParams(location.search).has('test')) game.setPaused(true);
  hud.setAssets(assets);
  hud.update(game);
  const renderer = createRenderer(root, hud.canvas, game, assets);
  hud.loading('Warming up the steering network…');
  const [x, y, theta] = axles(game.state)[2];
  await policy.predict(tokensFor(route, x, y, theta, 0), new Float32Array([0, 1, 0, 1]));
  hud.ready();
  hud.update(game);
  let stroke: Point[] = [],
    activePointer: number | undefined,
    previous = performance.now(),
    running = true;
  const reset = () => {
    if (game.drawing) toggleDrawing();
    game.reset();
    hud.canvas.focus({ preventScroll: true });
  };
  const toggleDrawing = () => {
    game.setDrawing(!game.drawing);
    stroke = [];
    hud.canvas.style.cursor = game.drawing ? 'crosshair' : 'default';
  };
  hud.get('reset').onclick = reset;
  hud.get('alert-reset').onclick = () => {
    if (game.status || game.failure) reset();
    game.setPaused(false);
  };
  hud.get('cancel').onclick = toggleDrawing;
  hud.get('speed').onclick = () => {
    game.speedMultiplier = speeds[(speeds.indexOf(game.speedMultiplier) + 1) % speeds.length];
  };
  hud.get('mode').onclick = () => game.switchMode();
  hud.get('pause').onclick = () => game.setPaused(!game.paused);
  hud.get('draw').onclick = toggleDrawing;
  hud.get('telemetry').onclick = () => {
    game.diagnostics = !game.diagnostics;
  };
  hud.get('rays').onclick = () => {
    game.rays = !game.rays;
  };
  hud.get('new').onclick = () => game.reset(sampleRoute());
  const speeds = [1, 2, 4, 8, 16];
  const keydown = (event: KeyboardEvent) => {
    const key = event.key.toLowerCase();
    if ([' ', 'arrowup', 'arrowdown', 'arrowleft', 'arrowright'].includes(key))
      event.preventDefault();
    if (event.target instanceof HTMLButtonElement && key === 'enter') return;
    game.keys.add(key);
    if (event.repeat) return;
    if (game.drawing) {
      if (key === 'escape') toggleDrawing();
      else if (key === 'v') game.rays = !game.rays;
      return;
    }
    if (key === 'escape' && game.drawing) toggleDrawing();
    else if (key === 'r') reset();
    else if (key === 'm') game.switchMode();
    else if (key === ' ') game.setPaused(!game.paused);
    else if (key === 't')
      game.speedMultiplier = speeds[(speeds.indexOf(game.speedMultiplier) + 1) % speeds.length];
    else if (key === 'e') toggleDrawing();
    else if (key === 'h') game.diagnostics = !game.diagnostics;
    else if (key === 'v') game.rays = !game.rays;
    else if (key === 'p') hud.get('new').click();
  };
  const keyup = (event: KeyboardEvent) => game.keys.delete(event.key.toLowerCase());
  const blur = () => {
    game.keys.clear();
    if (!game.paused && !game.drawing) game.setPaused(true);
  };
  window.addEventListener('keydown', keydown);
  window.addEventListener('keyup', keyup);
  window.addEventListener('blur', blur);
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) blur();
    previous = performance.now();
  });
  hud.canvas.onpointerdown = (event) => {
    hud.canvas.focus({ preventScroll: true });
    if (!game.drawing || event.button !== 0) return;
    activePointer = event.pointerId;
    hud.canvas.setPointerCapture(event.pointerId);
    stroke = [renderer.screenToWorld(event.clientX, event.clientY)];
  };
  hud.canvas.onpointermove = (event) => {
    if (activePointer !== event.pointerId) return;
    const p = renderer.screenToWorld(event.clientX, event.clientY),
      last = stroke.at(-1)!;
    if (Math.hypot(p[0] - last[0], p[1] - last[1]) > 0.4) stroke.push(p);
  };
  hud.canvas.onpointerup = (event) => {
    if (activePointer !== event.pointerId) return;
    activePointer = undefined;
    hud.canvas.releasePointerCapture(event.pointerId);
    if (!game.drawing) {
      stroke = [];
      return;
    }
    try {
      const next = drawnRoute(stroke);
      game.reset(next);
      game.setDrawing(false);
      stroke = [];
      hud.canvas.style.cursor = 'default';
    } catch (e) {
      stroke = [];
      hud.get('drawing-hint').querySelector('span')!.textContent =
        e instanceof Error ? e.message : String(e);
    }
  };
  hud.canvas.onpointercancel = () => {
    activePointer = undefined;
    stroke = [];
  };
  const animate = (now: number) => {
    if (!running) return;
    const elapsed = (now - previous) / 1000;
    previous = now;
    if (!new URLSearchParams(location.search).has('freeze')) void game.advance(elapsed);
    hud.update(game);
    renderer.render(now, stroke);
    requestAnimationFrame(animate);
  };
  requestAnimationFrame(animate);
  root.device.lost.then((info) => {
    if (running) {
      running = false;
      hud.failure(`Graphics device lost: ${info.message}. Reload to restart.`);
    }
  });
  window.addEventListener(
    'pagehide',
    () => {
      running = false;
      policy.dispose();
      renderer.dispose();
      root.destroy();
    },
    { once: true },
  );
  if (new URLSearchParams(location.search).has('test')) {
    const { attachHarness } = await import('./testHarness');
    attachHarness(game, policy, root);
  }
}
boot().catch((e) => {
  console.error(e);
  hud.failure(e instanceof Error ? e.message : String(e));
});
