/** Desktop depot placement and paint rules, using the same seeded scenery and snow assets. */
import { initialState, axles, type Point } from '../simulation/physics';
import type { Route } from '../simulation/path';
import { type Assets, type RGB, polygon, line, rect, fill, css, circle } from './assets';
export interface Camera {
  width: number;
  height: number;
  scale: number;
  project(x: number, y: number): Point;
}
export interface Prop {
  kind: string;
  x: number;
  y: number;
  length: number;
  width: number;
  heading: number;
  number: number;
}
export interface Dock {
  x: number;
  y: number;
  heading: number;
  warehouse: boolean;
}
export const boxPoints = (x: number, y: number, length: number, width: number, h = 0): Point[] =>
  [
    [-length / 2, -width / 2],
    [length / 2, -width / 2],
    [length / 2, width / 2],
    [-length / 2, width / 2],
  ].map(([a, b]) => [x + a * Math.cos(h) - b * Math.sin(h), y + a * Math.sin(h) + b * Math.cos(h)]);
const canvas = (width: number, height: number) => {
  const c = document.createElement('canvas');
  c.width = width;
  c.height = height;
  // These are CPU masks: placement and road-paint checks repeatedly read their pixels.
  c.getContext('2d', { willReadFrequently: true });
  return c;
};
function capsules(camera: Camera, points: Point[], radius: number) {
  const c = canvas(camera.width, camera.height),
    ctx = c.getContext('2d')!;
  ctx.fillStyle = 'white';
  ctx.strokeStyle = 'white';
  ctx.lineWidth = 2 * Math.max(1, Math.round(radius));
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  if (points.length > 1) {
    ctx.beginPath();
    ctx.moveTo(...points[0]);
    for (const p of points.slice(1)) ctx.lineTo(...p);
    ctx.stroke();
  }
  for (const p of points) {
    ctx.beginPath();
    ctx.arc(...p, Math.max(1, Math.round(radius)), 0, Math.PI * 2);
    ctx.fill();
  }
  return c;
}
export function configureDepot(route: Route): { props: Prop[]; dock: Dock } {
  const maskCamera: Camera = {
    width: 320,
    height: 200,
    scale: 2,
    project: (x, y) => [Math.round((x + 80) * 2), Math.round((50 - y) * 2)],
  };
  const routePixels = route.points.map((p) => maskCamera.project(...p)),
    reserved = capsules(maskCamera, routePixels, (Math.hypot(16.5, 1.25) + 6) * 2),
    ctx = reserved.getContext('2d')!;
  const start = initialState(route.points[0], route.headings[0]),
    [[x0, y0, h0], [x1, y1, h1], [x2, y2, h2]] = axles(start);
  const starting = canvas(320, 200),
    startCtx = starting.getContext('2d')!;
  for (const [x, y, h, length, width, offset] of [
    [x0, y0, h0, 5, 2.4, 2],
    [x1, y1, h1, 7.5, 2.5, 2.25],
    [x2, y2, h2, 7.5, 2.5, 2.25],
  ]) {
    const points = boxPoints(
      x + offset * Math.cos(h),
      y + offset * Math.sin(h),
      length,
      width,
      h,
    ).map((p) => maskCamera.project(...p));
    startCtx.drawImage(capsules(maskCamera, [...points, points[0]], 6), 0, 0);
    polygon(startCtx, points, [255, 255, 255]);
  }
  ctx.drawImage(starting, 0, 0);
  const [ex, ey] = route.points.at(-1)!,
    heading = route.headings.at(-1)!;
  const footprint = (p: Prop, margin = 0) => {
    const c = canvas(320, 200);
    polygon(
      c.getContext('2d')!,
      boxPoints(p.x, p.y, p.length + 2 * margin, p.width + 2 * margin, p.heading).map((point) =>
        maskCamera.project(...point),
      ),
      [255, 255, 255],
    );
    return c;
  };
  const inside = (p: Prop, margin = 0) =>
    boxPoints(p.x, p.y, p.length + 2 * margin, p.width + 2 * margin, p.heading).every(
      ([x, y]) => Math.abs(x) <= 79 && Math.abs(y) <= 49,
    );
  const overlap = (a: HTMLCanvasElement, b: HTMLCanvasElement) => {
    const x = a.getContext('2d')!.getImageData(0, 0, 320, 200).data,
      y = b.getContext('2d')!.getImageData(0, 0, 320, 200).data;
    for (let i = 3; i < x.length; i += 4) if (x[i] && y[i]) return true;
    return false;
  };
  ctx.drawImage(
    footprint({
      kind: 'bay',
      x: ex - 5 * Math.cos(heading),
      y: ey - 5 * Math.sin(heading),
      length: 18,
      width: 14,
      heading,
      number: 0,
    }),
    0,
    0,
  );
  const building: Prop = {
      kind: 'warehouse',
      x: ex + 5 * Math.cos(heading),
      y: ey + 5 * Math.sin(heading),
      length: 4,
      width: 14,
      heading,
      number: 1,
    },
    buildingMask = footprint(building, 1);
  const early = routePixels.filter(
      (_, i) => (i * route.length) / (routePixels.length - 1) < route.length - 12,
    ),
    earlyMask = capsules(maskCamera, early, 12);
  const dock: Dock = {
    x: ex,
    y: ey,
    heading,
    warehouse:
      Math.min(80 - Math.abs(ex), 50 - Math.abs(ey)) < 12 &&
      inside(building) &&
      !overlap(earlyMask, buildingMask) &&
      !overlap(starting, buildingMask),
  };
  const candidates: Prop[] = [];
  const add = (kind: string, x: number, y: number, length: number, width: number, heading = 0) =>
    candidates.push({ kind, x, y, length, width, heading, number: candidates.length + 1 });
  for (const y of [-43, 43]) for (const x of [-52, 0, 52]) add('warehouse', x, y, 32, 10);
  for (const x of [-73, 73])
    for (const y of [-31, -13, 5, 23]) add('container', x, y, 12, 3, Math.PI / 2);
  for (const y of [-30, 30]) for (const x of [-56, -28, 0, 28, 56]) add('trailer', x, y, 9, 2.6);
  for (const x of [-76, -38, 0, 38, 76]) for (const y of [-46, 46]) add('light', x, y, 2, 2);
  const props: Prop[] = [];
  for (const p of candidates) {
    const shape = footprint(p, 1);
    if (inside(p, 1) && !overlap(reserved, shape)) {
      props.push(p);
      ctx.drawImage(shape, 0, 0);
    }
  }
  return { props, dock };
}
function label(
  ctx: CanvasRenderingContext2D,
  camera: Camera,
  assets: Assets,
  value: string,
  point: Point,
  color: RGB,
  scale: number,
  heading = 0,
) {
  if (camera.scale < 3) return;
  const image = assets.glyph(value, `mono_bold:${Math.trunc(14 * scale)}`, color),
    [x, y] = camera.project(...point);
  let angle = (((heading * 180) / Math.PI + 90) % 180) - 90;
  ctx.save();
  ctx.translate(x, y);
  if (Math.abs(angle) > 3) ctx.rotate((-angle * Math.PI) / 180);
  ctx.drawImage(image, -Math.floor(image.width / 2), -Math.floor(image.height / 2));
  ctx.restore();
}
function drawProp(
  ctx: CanvasRenderingContext2D,
  camera: Camera,
  assets: Assets,
  p: Prop,
  scale: number,
) {
  const points = boxPoints(p.x, p.y, p.length, p.width, p.heading).map((point) =>
      camera.project(...point),
    ),
    px = camera.scale;
  polygon(
    ctx,
    points.map(([x, y]) => [
      x + Math.max(2, Math.round(px * 0.45)),
      y + Math.max(3, Math.round(px * 0.7)),
    ]),
    [16, 20, 26],
  );
  if (p.kind === 'light') {
    const center = camera.project(p.x, p.y),
      top: Point = [center[0] - 2, center[1] - Math.round(px * 1.3)];
    circle(ctx, center, Math.max(2, Math.round(px * 0.6)), [58, 62, 62]);
    line(ctx, center, top, [136, 136, 117], 2);
    circle(ctx, top, Math.max(2, Math.round(px * 0.22)), [255, 207, 117]);
    return;
  }
  const palette: RGB[] = [
      [57, 77, 83],
      [91, 78, 62],
      [62, 76, 67],
    ],
    roof: RGB = p.kind === 'warehouse' ? [48, 53, 61] : palette[p.number % 3],
    c = Math.cos(p.heading),
    s = Math.sin(p.heading);
  if (p.kind === 'trailer') {
    for (const axle of [-p.length * 0.3 - 0.45, -p.length * 0.3 + 0.45])
      for (const side of [-p.width * 0.54, p.width * 0.54])
        polygon(
          ctx,
          boxPoints(p.x + axle * c - side * s, p.y + axle * s + side * c, 0.8, 0.35, p.heading).map(
            (point) => camera.project(...point),
          ),
          [10, 14, 18],
        );
    polygon(
      ctx,
      boxPoints(
        p.x + (p.length / 2 + 0.25) * c,
        p.y + (p.length / 2 + 0.25) * s,
        0.5,
        p.width * 0.6,
        p.heading,
      ).map((point) => camera.project(...point)),
      [40, 47, 54],
    );
  }
  const lift = Math.max(2, Math.round(px * (p.kind === 'warehouse' ? 0.65 : 0.35))),
    top = points.map(([x, y]) => [x, y - lift] as Point);
  polygon(ctx, points, [26, 33, 40]);
  for (let i = 0; i < 4; i++) {
    const j = (i + 1) % 4;
    polygon(ctx, [points[i], points[j], top[j], top[i]], [36, 42, 49]);
  }
  polygon(ctx, top, roof);
  polygon(ctx, top, roof.map((v) => v + 20) as unknown as RGB, 1);
  for (let off = -p.length / 2 + 0.5; off < p.length / 2; off += p.kind === 'container' ? 0.7 : 2) {
    const ends = [-p.width * 0.42, p.width * 0.42].map((side) => {
      const q = camera.project(p.x + off * c - side * s, p.y + off * s + side * c);
      return [q[0], q[1] - lift] as Point;
    });
    line(ctx, ends[0], ends[1], roof.map((v) => v + 9) as unknown as RGB);
  }
  if (p.kind === 'warehouse') {
    const south = [...points]
      .sort((a, b) => b[1] - a[1])
      .slice(0, 2)
      .sort((a, b) => a[0] - b[0]);
    for (const off of p.length >= 24 ? [-9, 0, 9] : [0]) {
      const f = 0.5 + off / p.length,
        x = Math.round(south[0][0] + f * (south[1][0] - south[0][0])),
        y = Math.round(south[0][1] + f * (south[1][1] - south[0][1])),
        w = Math.max(3, Math.round(px * 3.6)),
        door = rect(x - Math.floor(w / 2), y - lift, w, lift);
      fill(ctx, door, [19, 25, 31]);
      polygon(
        ctx,
        [
          [door.x, door.y],
          [door.x + door.w - 1, door.y],
          [door.x + door.w - 1, door.y + door.h - 1],
          [door.x, door.y + door.h - 1],
        ],
        [126, 115, 74],
        1,
      );
    }
    label(
      ctx,
      camera,
      assets,
      `DEPOT ${String(p.number).padStart(2, '0')}`,
      [p.x, p.y + 2.7],
      [158, 160, 145],
      scale,
    );
  } else
    label(
      ctx,
      camera,
      assets,
      String(p.number).padStart(2, '0'),
      [p.x, p.y],
      [166, 174, 166],
      scale,
      p.heading,
    );
}
export function paintTerrain(
  ctx: CanvasRenderingContext2D,
  camera: Camera,
  assets: Assets,
  route: Route,
  editing: boolean,
  uiScale: number,
  depot: ReturnType<typeof configureDepot>,
) {
  const project = (point: Point) => camera.project(...point),
    px = camera.scale,
    { width, height } = camera;
  fill(ctx, rect(0, 0, width, height), [11, 15, 20]);
  for (const [x, y, color] of assets.scenery.slabs) {
    const points = boxPoints(x + 8, y + 5, 16, 10).map(project);
    polygon(ctx, points, color);
    polygon(ctx, points, [24, 29, 34], 1);
  }
  for (const [x, y, l, w, h] of assets.scenery.patches) {
    const points = boxPoints(x, y, l, w, h).map(project);
    polygon(ctx, points, [28, 33, 38]);
    polygon(ctx, points, [40, 44, 47], 1);
  }
  for (const [x, y, tone] of assets.scenery.grain) {
    const p = camera.project(x, y);
    fill(ctx, rect(...p, 1, 1), [tone, tone + 3, tone + 5]);
  }
  for (const [x, y, r] of assets.scenery.stains) {
    ctx.fillStyle = css([26, 30, 34]);
    ctx.beginPath();
    ctx.arc(...camera.project(x, y), Math.max(1, Math.round(r * px)), 0, Math.PI * 2);
    ctx.fill();
  }
  if (!editing) {
    const [x, y] = camera.project(-80, 50),
      w = Math.round(160 * px),
      h = Math.round(100 * px);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(assets.snowShadow, x, y, w, h);
    ctx.drawImage(assets.snow, x, y, w, h);
  }
  for (const y of [-47, 47]) {
    line(
      ctx,
      camera.project(-75, y),
      camera.project(75, y),
      [18, 23, 28],
      Math.max(2, Math.round(px * 0.35)),
    );
    for (let x = -75; x < 75; x += 1.2)
      line(ctx, camera.project(x, y - 0.16), camera.project(x, y + 0.16), [52, 57, 59]);
  }
  polygon(
    ctx,
    boxPoints(0, 0, 160, 100).map(project),
    [75, 82, 83],
    Math.max(2, Math.round(px * 0.45)),
  );
  polygon(ctx, boxPoints(0, 0, 158, 98).map(project), [53, 63, 68], 1);
  for (let y = -48; y < 50; y += 8)
    for (const x of [-80, 80]) {
      ctx.fillStyle = css([97, 105, 106]);
      ctx.beginPath();
      ctx.arc(...camera.project(x, y), Math.max(1, Math.round(px * 0.2)), 0, Math.PI * 2);
      ctx.fill();
    }
  label(ctx, camera, assets, 'NORTH APRON // 04', [0, 35], [76, 81, 82], uiScale);
  label(ctx, camera, assets, 'FREIGHT ONLY', [-52, -35], [76, 81, 82], uiScale);
  if (!editing) {
    for (const p of depot.props)
      if (p.kind === 'light') {
        for (const [r, alpha] of [
          [6, 8],
          [3, 12],
        ]) {
          ctx.fillStyle = css([236, 180, 83], alpha / 255);
          ctx.beginPath();
          ctx.arc(...camera.project(p.x, p.y), Math.max(2, Math.round(px * r)), 0, Math.PI * 2);
          ctx.fill();
        }
      }
    for (const p of depot.props) drawProp(ctx, camera, assets, p, uiScale);
  }
  if (!editing) {
    const points = route.points.map(project),
      colorMask = (mask: HTMLCanvasElement, color: RGB) => {
        const c = canvas(width, height),
          cctx = c.getContext('2d')!;
        cctx.drawImage(mask, 0, 0);
        cctx.globalCompositeOperation = 'source-in';
        cctx.fillStyle = css(color);
        cctx.fillRect(0, 0, width, height);
        ctx.drawImage(c, 0, 0);
      };
    colorMask(capsules(camera, points, 5.3 * px), [44, 47, 47]);
    const road = capsules(camera, points, 4.8 * px),
      inner = capsules(camera, points, 4.56 * px);
    colorMask(road, [19, 26, 32]);
    const edge = canvas(width, height),
      ec = edge.getContext('2d')!;
    ec.drawImage(road, 0, 0);
    ec.globalCompositeOperation = 'destination-out';
    ec.drawImage(inner, 0, 0);
    colorMask(edge, [222, 228, 226]);
    const mask = road.getContext('2d')!.getImageData(0, 0, width, height).data,
      isRoad = (p: Point) =>
        p[0] >= 0 &&
        p[1] >= 0 &&
        p[0] < width &&
        p[1] < height &&
        mask[(p[1] * width + p[0]) * 4 + 3] > 0;
    for (const [x, y] of assets.scenery.grain) {
      const p = camera.project(x, y);
      if (isRoad(p)) fill(ctx, rect(...p, 1, 1), [25, 32, 38]);
    }
    for (const side of [-1.25, 1.25])
      for (let i = 1; i < route.points.length; i++) {
        const a = route.points[i - 1],
          b = route.points[i],
          ha = route.headings[i - 1],
          hb = route.headings[i];
        line(
          ctx,
          camera.project(a[0] - Math.sin(ha) * side, a[1] + Math.cos(ha) * side),
          camera.project(b[0] - Math.sin(hb) * side, b[1] + Math.cos(hb) * side),
          [14, 20, 25],
          Math.max(1, Math.round(px * 0.28)),
        );
      }
    for (let i = 0; i < route.points.length - 1; i++) {
      const [x, y] = route.points[i],
        h = route.headings[i],
        s = (i * route.length) / (route.points.length - 1);
      if (Math.floor(s / 2.4) % 2 === 0)
        line(ctx, points[i], points[i + 1], [222, 228, 226], Math.max(1, Math.round(px * 0.15)));
      if (Math.abs(route.curvatures[i]) > 0.055 && i % 3 === 0) {
        const side = -Math.sign(route.curvatures[i]) * 5.05,
          cx = x - Math.sin(h) * side,
          cy = y + Math.cos(h) * side,
          p = camera.project(cx, cy);
        if (!isRoad(p)) polygon(ctx, boxPoints(cx, cy, 1.2, 0.38, h).map(project), [196, 156, 64]);
      }
    }
    for (let distance = 13; distance < route.length - 10; distance += 28) {
      const i = Math.min(
          route.points.length - 1,
          Math.ceil((distance / route.length) * (route.points.length - 1)),
        ),
        [x, y] = route.points[i],
        h = route.headings[i],
        c = Math.cos(h),
        s = Math.sin(h);
      polygon(
        ctx,
        [
          [x + 1.2 * c, y + 1.2 * s],
          [x - 0.8 * c - 0.7 * s, y - 0.8 * s + 0.7 * c],
          [x - 0.3 * c, y - 0.3 * s],
          [x - 0.8 * c + 0.7 * s, y - 0.8 * s - 0.7 * c],
        ].map((p) => camera.project(p[0], p[1])),
        [222, 228, 226],
      );
    }
    const { dock } = depot,
      c = Math.cos(dock.heading),
      s = Math.sin(dock.heading),
      point = (a: number, b: number): Point => [dock.x + a * c - b * s, dock.y + a * s + b * c],
      frame = [point(-11, -4.5), point(0, -4.5), point(0, 4.5), point(-11, 4.5)];
    for (const [a, b] of [
      [frame[0], frame[1]],
      [frame[1], frame[2]],
      [frame[2], frame[3]],
    ])
      line(ctx, project(a), project(b), [222, 228, 226], Math.max(1, Math.round(px * 0.2)));
    for (const side of [-1, 1]) {
      for (let a = -10; a < -1; a += 2)
        line(
          ctx,
          project(point(a, side * 4.5)),
          project(point(a + 1, side * 3.8)),
          [179, 188, 186],
          Math.max(1, Math.round(px * 0.14)),
        );
      line(
        ctx,
        project(point(1.7, side * 3.1)),
        project(point(1.7, side * 4.2)),
        [10, 14, 18],
        Math.max(2, Math.round(px * 0.5)),
      );
    }
    label(ctx, camera, assets, 'BAY 01', point(-5.5, 5.8), [222, 228, 226], uiScale, dock.heading);
    if (dock.warehouse)
      drawProp(
        ctx,
        camera,
        assets,
        {
          kind: 'warehouse',
          x: point(5, 0)[0],
          y: point(5, 0)[1],
          length: 4,
          width: 14,
          heading: dock.heading,
          number: 1,
        },
        uiScale,
      );
  }
  // The desktop vignette consists of four discrete rings, not a smooth gradient.
  const diagonal = Math.hypot(width, height);
  for (const [fraction, alpha] of [
    [0.7, 35],
    [0.8, 80],
    [0.9, 140],
    [1.02, 210],
  ]) {
    const radius = Math.trunc(diagonal * 0.5 * fraction),
      thickness = Math.trunc(diagonal * 0.12);
    ctx.beginPath();
    ctx.arc(Math.floor(width / 2), Math.floor(height / 2), radius - thickness / 2, 0, Math.PI * 2);
    ctx.strokeStyle = css([8, 10, 14], alpha / 255);
    ctx.lineWidth = thickness;
    ctx.stroke();
  }
}
