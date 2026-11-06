/** Desktop layer order: baked depot, lighting/previews, TypeGPU truck, then snowfall. */
import type { TgpuRoot } from 'typegpu';
import type { Game } from '../simulation/game';
import { axles, clamp, step, type Point } from '../simulation/physics';
import { type Assets, css, polygon, line, rect, fill } from './assets';
import { type Camera, boxPoints, configureDepot, paintTerrain } from './terrain';
import { Mesh, createPainter } from './primitives';
import { vehicle } from './vehicle';
import { createAttentionLayer } from './attention';
export function createRenderer(
  root: TgpuRoot,
  canvas: HTMLCanvasElement,
  game: Game,
  assets: Assets,
) {
  const attentionLayer = createAttentionLayer();
  const painter = createPainter(root, canvas),
    host = canvas.parentElement!;
  const layer = (id: string, z: string) => {
    const c = document.createElement('canvas');
    c.id = id;
    Object.assign(c.style, { zIndex: z, pointerEvents: 'none' });
    host.append(c);
    return c;
  };
  const ground = layer('ground', '0'),
    alpha = layer('lighting', '1'),
    weather = layer('weather', '3');
  canvas.style.zIndex = '2';
  const back = ground.getContext('2d')!,
    dynamic = alpha.getContext('2d')!,
    snow = weather.getContext('2d')!;
  let camera: Camera = { width: 1, height: 1, scale: 1, project: () => [0, 0] },
    dirty = true,
    route = game.route,
    editing = game.drawing,
    elapsed = 0,
    last = 0;
  let depot = configureDepot(route);
  const backgrounds = new Map<string, HTMLCanvasElement>();
  const resize = () => {
    const r = canvas.getBoundingClientRect(),
      width = Math.max(1, Math.round(r.width)),
      height = Math.max(1, Math.round(r.height));
    if (camera.width === width && camera.height === height) return;
    const scale = Math.min(Math.max(1, width - 48) / 160, Math.max(1, height - 48) / 100);
    camera = {
      width,
      height,
      scale,
      project: (x, y) => [
        Math.trunc(Math.floor(width / 2) + x * scale),
        Math.trunc(Math.floor(height / 2) - y * scale),
      ],
    };
    canvas.width = width;
    canvas.height = height;
    for (const c of [ground, alpha, weather]) {
      c.width = width;
      c.height = height;
      Object.assign(c.style, {
        left: canvas.style.left,
        top: canvas.style.top,
        width: `${width}px`,
        height: `${height}px`,
      });
      c.getContext('2d')!.imageSmoothingEnabled = false;
    }
    dirty = true;
  };
  const observer = new ResizeObserver(resize);
  observer.observe(canvas);
  const sprite = (name: string, point: Point, heading = 0) => {
    const source = assets.lightImages[name],
      spec = assets.scenery.lights[name],
      factor = camera.scale / 8;
    dynamic.save();
    dynamic.translate(...camera.project(...point));
    dynamic.rotate(-heading);
    dynamic.drawImage(
      source,
      -spec.origin[0] * factor,
      -spec.origin[1] * factor,
      spec.size[0] * factor,
      spec.size[1] * factor,
    );
    dynamic.restore();
  };
  const shadow = (state: Game['state']) => {
    const [[x0, y0, h0], [x1, y1, h1], [x2, y2, h2]] = axles(state),
      bodies = [
        [x0, y0, h0, 1.9, 5.3, 2.4],
        [x0, y0, h1, -2.4, 4.1, 2.5],
        [x2, y2, h2, 3, 7.7, 2.5],
      ];
    for (const [spread, opacity] of [
      [0.65, 16],
      [0.42, 28],
      [0.23, 45],
      [0, 76],
    ])
      for (const [x, y, h, offset, length, width] of bodies)
        polygon(
          dynamic,
          boxPoints(
            x + offset * Math.cos(h) + 0.55,
            y + offset * Math.sin(h) - 0.75,
            length + 2 * spread,
            width + 2 * spread,
            h,
          ).map((p) => camera.project(...p)),
          [4, 8, 13],
          0,
          opacity / 255,
        );
    for (const [x, y, h, width] of [
      [x0, y0, h0, 2.4],
      [x1, y1, h1, 2.5],
      [x2, y2, h2, 2.5],
    ])
      for (const offset of [-0.58, 0.58])
        for (const side of [-width * 0.47, width * 0.47]) {
          const p = camera.project(
            x + offset * Math.cos(h) - side * Math.sin(h),
            y + offset * Math.sin(h) + side * Math.cos(h),
          );
          dynamic.fillStyle = css([3, 6, 10], 145 / 255);
          dynamic.beginPath();
          dynamic.ellipse(
            p[0],
            p[1],
            Math.max(3, Math.round(camera.scale * 1.05)) / 2,
            Math.max(2, Math.round(camera.scale * 0.55)) / 2,
            0,
            0,
            Math.PI * 2,
          );
          dynamic.fill();
        }
  };
  const dockLights = (ctx: CanvasRenderingContext2D, glow: boolean) => {
    const { dock } = depot,
      color =
        game.status === 'DOCK SECURED' ? ([105, 235, 157] as const) : ([245, 113, 68] as const),
      r = Math.max(3, Math.round(camera.scale * 0.45));
    for (const side of [-4.7, 4.7]) {
      const p = camera.project(
        dock.x - Math.sin(dock.heading) * side,
        dock.y + Math.cos(dock.heading) * side,
      );
      if (glow) {
        ctx.fillStyle = css(color, 35 / 255);
        ctx.beginPath();
        ctx.arc(...p, r * 3, 0, Math.PI * 2);
        ctx.fill();
      } else {
        fill(ctx, rect(p[0] - r - 2, p[1] - r - 2, 2 * r + 4, 2 * r + 4), [12, 17, 22]);
        fill(ctx, rect(p[0] - r, p[1] - r, 2 * r + 1, 2 * r + 1), color);
      }
    }
  };
  return {
    screenToWorld(x: number, y: number): Point {
      const r = canvas.getBoundingClientRect();
      return [
        clamp((x - r.left - Math.floor(camera.width / 2)) / camera.scale, -72, 72),
        clamp(-(y - r.top - Math.floor(camera.height / 2)) / camera.scale, -42, 42),
      ];
    },
    render(now: number, stroke: Point[]) {
      resize();
      for (const c of [ground, alpha, weather]) {
        c.style.left = canvas.style.left;
        c.style.top = canvas.style.top;
      }
      if (game.route !== route) {
        route = game.route;
        depot = configureDepot(route);
        backgrounds.clear();
        dirty = true;
      }
      if (game.drawing !== editing) {
        editing = game.drawing;
        dirty = true;
      }
      if (dirty) {
        const uiScale = Math.max(1, Math.min(innerWidth / 1400, innerHeight / 900));
        const key = `${camera.width}/${camera.height}/${editing}/${uiScale}`;
        const cached = backgrounds.get(key);
        if (cached) back.drawImage(cached, 0, 0);
        else {
          paintTerrain(back, camera, assets, route, editing, uiScale, depot);
          const snapshot = document.createElement('canvas');
          snapshot.width = camera.width;
          snapshot.height = camera.height;
          snapshot.getContext('2d')!.drawImage(ground, 0, 0);
          backgrounds.set(key, snapshot);
          if (backgrounds.size > 2) backgrounds.delete(backgrounds.keys().next().value!);
        }
        dirty = false;
      }
      dynamic.clearRect(0, 0, camera.width, camera.height);
      snow.clearRect(0, 0, camera.width, camera.height);
      const mesh = new Mesh(camera.scale, camera.width, camera.height);
      if (!editing) {
        dockLights(dynamic, true);
        shadow(game.state);
        const [[x0, y0, h0], , [x2, y2, h2]] = axles(game.state),
          c = Math.cos(h0),
          s = Math.sin(h0);
        for (const side of [-2.4 * 0.72 * 0.46, 2.4 * 0.72 * 0.46]) {
          const point: Point = [x0 + 4.45 * c - side * s, y0 + 4.45 * s + side * c];
          sprite('head', point, h0);
          sprite('white', point);
        }
        const offset = 6 + 0.85 - 7.7 - 0.08;
        if (game.speed < -0.1)
          for (const side of [-2.5 * 0.27, 2.5 * 0.27])
            sprite(
              'reverse',
              [
                x2 + offset * Math.cos(h2) - side * Math.sin(h2),
                y2 + offset * Math.sin(h2) + side * Math.cos(h2),
              ],
              h2 + Math.PI,
            );
        for (const side of [-2.5 * 0.44, 2.5 * 0.44])
          sprite('red', [
            x2 + offset * Math.cos(h2) - side * Math.sin(h2),
            y2 + offset * Math.sin(h2) + side * Math.cos(h2),
          ]);
        if (
          game.mode === 'MANUAL' &&
          game.status !== 'OFF TRACK' &&
          game.status !== 'DOCK SECURED'
        ) {
          let future = game.state;
          for (let i = 1; i <= 15; i++) {
            future = step(
              future,
              Math.abs(game.speed) > 0.1 ? game.speed : -1.5,
              game.targetSteer,
              0.2,
            );
            if (i % 3 === 0) {
              const [[x0, y0, h0], [x1, y1, h1], [x2, y2, h2]] = axles(future);
              for (const [x, y, h, len, width, off] of [
                [x0, y0, h0, 5, 2.4, 2],
                [x1, y1, h1, 7.5, 2.5, 2.25],
                [x2, y2, h2, 7.5, 2.5, 2.25],
              ])
                polygon(
                  dynamic,
                  boxPoints(x + off * Math.cos(h), y + off * Math.sin(h), len, width, h).map((p) =>
                    camera.project(...p),
                  ),
                  [255, 255, 255],
                  1,
                  Math.trunc(30 + (i / 15) * 85) / 255,
                );
            }
          }
        }
        if (game.mode === 'AUTONOMOUS' && game.rays) {
          const from = camera.project(x0 + 3.5 * 0.4 * c, y0 + 3.5 * 0.4 * s);
          const peak = Math.max(...game.attention, 0.001);
          const rays = game.attention.flatMap((value, i) => {
            const weight = value / peak;
            return weight > 0.15
              ? [
                  {
                    point: camera.project(
                      ...route.points[Math.min(route.points.length - 1, game.index + i)],
                    ),
                    weight,
                  },
                ]
              : [];
          });
          attentionLayer.draw(dynamic, from, rays);
        }
        dockLights(dynamic, false);
        vehicle(mesh, game.state);
        if (last && !new URLSearchParams(location.search).has('freeze'))
          elapsed += Math.min(0.1, Math.max(0, (now - last) / 1000));
        for (const [u, v, near, phase] of assets.scenery.flakes) {
          const x =
              (u * camera.width + elapsed * (5 + near * 9) + Math.sin(elapsed * 0.55 + phase) * 9) %
              Math.max(1, camera.width),
            y = (v * camera.height + elapsed * (10 + near * 20)) % Math.max(1, camera.height);
          fill(
            snow,
            rect(Math.round(x), Math.round(y), near < 0.6 ? 1 : 2, near < 0.6 ? 1 : 2),
            near < 0.6 ? [119, 136, 145] : near < 0.83 ? [164, 180, 186] : [205, 215, 215],
          );
        }
      } else
        for (let i = 1; i < stroke.length; i++)
          line(
            dynamic,
            camera.project(...stroke[i - 1]),
            camera.project(...stroke[i]),
            [255, 190, 73],
            2,
          );
      last = now;
      painter.draw(mesh);
    },
    dispose() {
      observer.disconnect();
      painter.dispose();
      ground.remove();
      alpha.remove();
      weather.remove();
    },
  };
}
