/** One-to-one geometry, colour, face ordering and pixel strokes from game/vehicle.py. */
import { axles, type Point, type State } from '../simulation/physics';
import { roundEven } from '../simulation/path';
import { Mesh, type Color } from './primitives';
type Point3 = [number, number, number];
interface Face {
  depth: number;
  points: Point3[];
  color: Color;
  outline?: Color;
  width: number;
}
const local = (x: number, y: number, h: number, a: number, b: number, z: number): Point3 => [
  x + a * Math.cos(h) - b * Math.sin(h),
  y + a * Math.sin(h) + b * Math.cos(h),
  z,
];
const project = ([x, y, z]: Point3): Point => [x - 0.1 * z, y + 0.3 * z];
const tint = (color: Color, multiplier: number, offset = 0): Color => [
  Math.min(255, roundEven(color[0] * multiplier + offset)),
  Math.min(255, roundEven(color[1] * multiplier + offset)),
  Math.min(255, roundEven(color[2] * multiplier + offset)),
];
export function vehicle(mesh: Mesh, state: State) {
  const faces: Face[] = [],
    scale = mesh.scale;
  const depth = (points: Point3[]) =>
    points.reduce((sum, [x, y, z]) => sum + 0.1 * x - 0.3 * y + z, 0) / points.length;
  const polygon = (points: Point3[], color: Color, outline?: Color) =>
    faces.push({ depth: depth(points), points, color, outline, width: 0 });
  const line = (points: Point3[], color: Color, width = 1) =>
    faces.push({ depth: depth(points) + 0.002, points, color, width });
  const box = (
    cx: number,
    cy: number,
    h: number,
    length: number,
    width: number,
    bottom: number,
    top: number,
    color: Color,
    details = false,
    chamfer = 0,
  ) => {
    const p = (x: number, y: number, z: number) => local(cx, cy, h, x, y, z),
      hl = length / 2,
      hw = width / 2;
    const corners: Point[] = chamfer
      ? [
          [-hl, -hw],
          [hl - chamfer, -hw],
          [hl, -hw + chamfer],
          [hl, hw - chamfer],
          [hl - chamfer, hw],
          [-hl, hw],
        ]
      : [
          [-hl, -hw],
          [hl, -hw],
          [hl, hw],
          [-hl, hw],
        ];
    const bevel = Math.min(0.14, (top - bottom) * 0.2, width * 0.08),
      shoulder = top - bevel;
    const inset: Point[] = corners.map(([x, y]) => [
      x - (x < 0 ? -bevel : bevel),
      y - (y < 0 ? -bevel : bevel),
    ]);
    polygon(
      inset.map(([x, y]) => p(x, y, top)),
      tint(color, 1.08, 8),
    );
    for (let i = 0; i < corners.length; i++) {
      const j = (i + 1) % corners.length,
        [ax, ay] = corners[i],
        [bx, by] = corners[j];
      polygon(
        [p(ax, ay, shoulder), p(bx, by, shoulder), p(...inset[j], top), p(...inset[i], top)],
        tint(color, 1.02, 4),
      );
    }
    for (let i = 0; i < corners.length; i++) {
      const [ax, ay] = corners[i],
        [bx, by] = corners[(i + 1) % corners.length],
        nx = by - ay,
        ny = ax - bx,
        c = Math.cos(h),
        s = Math.sin(h),
        wx = nx * c - ny * s,
        wy = nx * s + ny * c;
      if (wx * 0.1 - wy * 0.3 <= 0) continue;
      const shade = 0.72 + 0.2 * Math.max(0, (-0.6 * wx - 0.8 * wy) / Math.hypot(wx, wy)),
        wall = tint(color, shade);
      polygon(
        [p(ax, ay, bottom), p(bx, by, bottom), p(bx, by, shoulder), p(ax, ay, shoulder)],
        wall,
        tint(wall, 0.8),
      );
      if (details) {
        for (const z of [bottom + 0.12, top - 0.12])
          line([p(ax, ay, z), p(bx, by, z)], [109, 139, 145]);
        if (Math.abs(ax - bx) > length * 0.8) {
          const count = Math.max(2, roundEven(length / 0.6));
          for (let j = 1; j < count; j++) {
            const t = j / count,
              x = ax + (bx - ax) * t,
              y = ay + (by - ay) * t;
            line([p(x, y, bottom + 0.2), p(x, y, top - 0.2)], tint(wall, 1, 9));
          }
          line(
            [p(ax, ay, bottom + 0.48), p(bx, by, bottom + 0.48)],
            [195, 205, 194],
            Math.max(1, roundEven(scale * 0.09)),
          );
          for (const t of [0.08, 0.5, 0.92]) {
            const x = ax + (bx - ax) * t,
              y = ay + (by - ay) * t;
            line([p(x - 0.07, y, bottom + 0.23), p(x + 0.07, y, bottom + 0.23)], [255, 179, 58], 2);
          }
        } else if (ax < 0 && bx < 0) {
          line([p(ax, 0, bottom + 0.16), p(ax, 0, top - 0.16)], [18, 31, 36]);
          for (const y of [-width * 0.23, width * 0.23]) {
            line([p(ax, y, bottom + 0.26), p(ax, y, top - 0.26)], [199, 211, 211]);
            for (const z of [bottom + 0.5, top - 0.5])
              line([p(ax, y - 0.11, z), p(ax, y + 0.11, z)], [223, 225, 213]);
          }
        }
      }
    }
    if (details) {
      const count = Math.max(2, roundEven(length / 0.9));
      for (let j = 1; j < count; j++) {
        const x = -length / 2 + (j * length) / count;
        line([p(x, -width * 0.43, top + 0.01), p(x, width * 0.43, top + 0.01)], tint(color, 1, 14));
      }
      for (const y of [-width * 0.46, width * 0.46])
        line(
          [p(-length / 2, y, top + 0.01), p(length / 2 - chamfer, y, top + 0.01)],
          [149, 175, 175],
        );
    }
  };
  const wheel = (cx: number, cy: number, h: number) => {
    const c = Math.cos(h),
      s = Math.sin(h);
    for (const side of [-1, 1]) {
      const ny = side * 0.19;
      if ((-s * 0.1 - c * 0.3) * side <= 0) continue;
      const ring = (r: number) =>
        Array.from({ length: 12 }, (_, j) => {
          const a = (j * (2 * Math.PI)) / 12;
          return local(cx, cy, h, r * Math.cos(a), ny, 0.48 + r * Math.sin(a));
        });
      polygon(ring(0.47), [13, 17, 22], [48, 53, 58]);
      polygon(ring(0.25), [122, 136, 144], [190, 199, 198]);
      polygon(ring(0.1), [38, 49, 57]);
    }
    box(cx, cy, h, 0.8, 0.36, 0.15, 0.75, [24, 28, 32]);
  };
  const [[x0, y0, h0], [x1, y1, h1], [x2, y2, h2]] = axles(state);
  const localBox = (
    x: number,
    y: number,
    h: number,
    offset: number,
    length: number,
    width: number,
    bottom: number,
    top: number,
    color: Color,
    details = false,
    chamfer = 0,
  ) => {
    const [cx, cy] = local(x, y, h, offset, 0, 0);
    box(cx, cy, h, length, width, bottom, top, color, details, chamfer);
  };
  localBox(x0, y0, h0, 1.4, 6, 2.4 * 0.65, 0.4, 0.82, [49, 58, 66]);
  localBox(x1, y1, h1, 2.65, 7.7, 2.5 * 0.67, 0.52, 0.91, [44, 56, 64]);
  localBox(x2, y2, h2, 2.4, 8.1, 2.5 * 0.67, 0.52, 0.91, [44, 56, 64]);
  for (const [x, y, h, width, offsets] of [
    [x0, y0, h0, 2.4, [-1.05, 0]],
    [x1, y1, h1, 2.5, [-0.58, 0.58]],
    [x2, y2, h2, 2.5, [-0.58, 0.58]],
  ] as [number, number, number, number, number[]][]) {
    for (const offset of offsets)
      for (const side of [-width * 0.47, width * 0.47]) {
        const [wx, wy] = local(x, y, h, offset, side, 0);
        wheel(wx, wy, h);
      }
  }
  for (const side of [-2.4 * 0.5, 2.4 * 0.5]) {
    const [wx, wy] = local(x0, y0, h0, 3.5, side, 0);
    wheel(wx, wy, h0 + state.phi);
  }
  localBox(x0, y0, h1, -2.4, 4.1, 2.5, 1.02, 3.5, [73, 126, 164], true, 0.8);
  localBox(x2, y2, h2, 6 + 0.85 - 7.7 / 2, 7.7, 2.5, 1.02, 3.5, [72, 143, 125], true);
  for (const [x, y, h] of [
    [x0, y0, h0],
    [x1, y1, h1],
  ])
    localBox(x, y, h, 0, 0.85, 0.9, 0.9, 1.05, [132, 144, 149]);
  localBox(x0, y0, h0, 1.65, 1.95, 2.4, 0.85, 2.95, [166, 40, 42]);
  localBox(x0, y0, h0, 1.45, 1.25, 2.4 * 0.76, 2.95, 3.12, [195, 56, 53]);
  const cp = (x: number, y: number, z: number) => local(x0, y0, h0, x, y, z);
  polygon(
    [cp(2.62, -1.04, 2.85), cp(2.62, 1.04, 2.85), cp(2.95, 1, 1.65), cp(2.95, -1, 1.65)],
    [34, 64, 79],
    [146, 176, 183],
  );
  line([cp(2.64, 0, 2.8), cp(2.93, 0, 1.7)], [14, 27, 35]);
  for (const side of [-1, 1]) {
    if (-Math.sin(h0) * side * 0.1 - Math.cos(h0) * side * 0.3 > 0) {
      const y = (side * 2.4) / 2;
      polygon(
        [cp(1.7, y, 2.7), cp(2.55, y, 2.7), cp(2.55, y, 1.85), cp(1.7, y, 1.85)],
        [28, 52, 65],
        [168, 181, 173],
      );
      line([cp(1.86, y, 1.62), cp(2.15, y, 1.62)], [213, 222, 215]);
    }
    const [tx, ty] = cp(0.9, side * 1.05, 0.3);
    box(tx, ty, h0, 1.45, 0.35, 0.35, 0.85, [153, 166, 169]);
    const [mx, my] = cp(2.6, side * 1.45, 2);
    box(mx, my, h0, 0.38, 0.18, 1.9, 2.45, [67, 80, 86]);
    line([cp(2.55, side * 1.1, 2.2), cp(2.6, side * 1.45, 2.2)], [172, 181, 169]);
  }
  localBox(x0, y0, h0, 3.65, 1.7, 2.4 * 0.76, 0.8, 1.65, [155, 34, 37]);
  line([cp(2.85, 0, 1.66), cp(4.45, 0, 1.66)], [219, 87, 77]);
  localBox(x0, y0, h0, 4.5, 0.16, 2.4 * 0.94, 0.55, 0.86, [184, 196, 196]);
  polygon(
    [cp(4.51, -0.57, 0.9), cp(4.51, 0.57, 0.9), cp(4.51, 0.57, 1.55), cp(4.51, -0.57, 1.55)],
    [31, 42, 50],
    [185, 198, 194],
  );
  for (const z of [1.03, 1.2, 1.37]) line([cp(4.52, -0.5, z), cp(4.52, 0.5, z)], [124, 143, 149]);
  for (const side of [-0.78, 0.78])
    polygon(
      [
        cp(4.53, side - 0.13, 0.96),
        cp(4.53, side + 0.13, 0.96),
        cp(4.53, side + 0.13, 1.25),
        cp(4.53, side - 0.13, 1.25),
      ],
      [255, 242, 204],
      [77, 88, 87],
    );
  const [ex, ey] = cp(0.5, -0.94, 0);
  box(ex, ey, h0, 0.18, 0.18, 0.85, 3.35, [157, 170, 176]);
  box(ex, ey, h0, 0.22, 0.22, 3.35, 3.4, [30, 39, 44]);
  const rear = 6 + 0.85 - 7.7 - 0.08;
  localBox(x2, y2, h2, rear, 0.18, 2.5, 0.42, 0.85, [78, 91, 96]);
  for (const [side, color] of [
    [-1.08, [242, 49, 49]],
    [-0.67, [238, 232, 206]],
    [0.67, [238, 232, 206]],
    [1.08, [242, 49, 49]],
  ] as [number, Color][]) {
    const rp = (y: number, z: number) => local(x2, y2, h2, rear - 0.1, y, z);
    polygon(
      [rp(side - 0.12, 0.53), rp(side + 0.12, 0.53), rp(side + 0.12, 0.77), rp(side - 0.12, 0.77)],
      color,
      [28, 35, 39],
    );
  }
  for (const face of faces.sort((a, b) => Math.round(a.depth * 1e9) - Math.round(b.depth * 1e9))) {
    const points = face.points.map(project);
    if (face.width) {
      for (let i = 1; i < points.length; i++)
        mesh.line(points[i - 1], points[i], face.width, face.color);
    } else {
      mesh.polygon(points, face.color);
      if (face.outline)
        for (let i = 0; i < points.length; i++)
          mesh.line(points[i], points[(i + 1) % points.length], 1, face.outline);
    }
  }
}
