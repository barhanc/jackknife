/** Low polygon miniature with the desktop truck's shallow oblique height axis. */
import { axles, type Point, type State } from '../simulation/physics';
import { Mesh, type Color } from './primitives';
type Point3 = [number, number, number];
const local = (x: number, y: number, h: number, a: number, b: number, z: number): Point3 => [
  x + a * Math.cos(h) - b * Math.sin(h),
  y + a * Math.sin(h) + b * Math.cos(h),
  z,
];
const project = ([x, y, z]: Point3): Point => [x - 0.1 * z, y + 0.3 * z];
export function vehicle(mesh: Mesh, state: State) {
  const [[x0, y0, h0], [x1, y1, h1], [x2, y2, h2]] = axles(state);
  const faces: { depth: number; points: Point3[]; color: Color; width: number }[] = [];
  const face = (points: Point3[], color: Color, width = 0) =>
    faces.push({
      depth:
        points.reduce((sum, [x, y, z]) => sum + 0.1 * x - 0.3 * y + z, 0) / points.length +
        (width ? 0.002 : 0),
      points,
      color,
      width,
    });
  const box = (
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
    const center = local(x, y, h, offset, 0, 0);
    const p = (a: number, b: number, z: number) => local(center[0], center[1], h, a, b, z);
    const hl = length / 2,
      hw = width / 2,
      bevel = Math.min(0.14, (top - bottom) * 0.2, width * 0.08),
      shoulder = top - bevel;
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
    const inset: Point[] = corners.map(([a, b]) => [
      a - Math.sign(a) * bevel,
      b - Math.sign(b) * bevel,
    ]);
    const tint = (scale: number, light = 0): Color => [
      Math.min(255, color[0] * scale + light),
      Math.min(255, color[1] * scale + light),
      Math.min(255, color[2] * scale + light),
    ];
    face(
      inset.map(([a, b]) => p(a, b, top)),
      tint(1.08, 8),
    );
    for (let i = 0; i < corners.length; i++) {
      const j = (i + 1) % corners.length,
        [a, b] = corners[i],
        [c, e] = corners[j];
      face(
        [p(a, b, shoulder), p(c, e, shoulder), p(...inset[j], top), p(...inset[i], top)],
        tint(1.02, 4),
      );
      const nx = e - b,
        ny = a - c,
        wx = nx * Math.cos(h) - ny * Math.sin(h),
        wy = nx * Math.sin(h) + ny * Math.cos(h);
      if (wx * 0.1 - wy * 0.3 <= 0) continue;
      const shade = 0.72 + 0.2 * Math.max(0, (-0.6 * wx - 0.8 * wy) / Math.hypot(wx, wy));
      face([p(a, b, bottom), p(c, e, bottom), p(c, e, shoulder), p(a, b, shoulder)], tint(shade));
      if (details) {
        for (const z of [bottom + 0.12, top - 0.12])
          face([p(a, b, z), p(c, e, z)], [109, 139, 145], 0.07);
        if (Math.abs(a - c) > length * 0.8) {
          const count = Math.max(2, Math.round(length / 0.6));
          for (let k = 1; k < count; k++) {
            const t = k / count;
            face(
              [
                p(a + (c - a) * t, b + (e - b) * t, bottom + 0.2),
                p(a + (c - a) * t, b + (e - b) * t, top - 0.2),
              ],
              tint(shade, 9),
              0.055,
            );
          }
          face([p(a, b, bottom + 0.48), p(c, e, bottom + 0.48)], [195, 205, 194], 0.09);
        } else if (a < 0 && c < 0) {
          face([p(a, 0, bottom + 0.16), p(a, 0, top - 0.16)], [18, 31, 36], 0.07);
          for (const side of [-0.23, 0.23])
            face(
              [p(a, width * side, bottom + 0.26), p(a, width * side, top - 0.26)],
              [199, 211, 211],
              0.07,
            );
        }
      }
    }
    if (details) {
      const count = Math.max(2, Math.round(length / 0.9));
      for (let i = 1; i < count; i++)
        face(
          [
            p(-hl + (i * length) / count, -width * 0.43, top + 0.01),
            p(-hl + (i * length) / count, width * 0.43, top + 0.01),
          ],
          tint(1, 14),
          0.05,
        );
      for (const side of [-0.46, 0.46])
        face(
          [p(-hl, width * side, top + 0.01), p(hl - chamfer, width * side, top + 0.01)],
          [149, 175, 175],
          0.06,
        );
    }
  };
  box(x0, y0, h0, 1.4, 6, 1.56, 0.4, 0.82, [49, 58, 66]);
  box(x1, y1, h1, 2.65, 7.7, 1.675, 0.52, 0.91, [44, 56, 64]);
  box(x2, y2, h2, 2.4, 8.1, 1.675, 0.52, 0.91, [44, 56, 64]);
  const wheel = (x: number, y: number, h: number) => {
    box(x, y, h, 0, 0.8, 0.36, 0.15, 0.75, [24, 28, 32]);
    for (const side of [-1, 1]) {
      if ((-Math.sin(h) * 0.1 - Math.cos(h) * 0.3) * side <= 0) continue;
      for (const [radius, color] of [
        [0.47, [13, 17, 22]],
        [0.25, [122, 136, 144]],
        [0.1, [38, 49, 57]],
      ] as [number, Color][]) {
        const points = Array.from({ length: 12 }, (_, j) =>
          local(
            x,
            y,
            h,
            radius * Math.cos((j * Math.PI) / 6),
            side * 0.19,
            0.48 + radius * Math.sin((j * Math.PI) / 6),
          ),
        );
        face(points, color);
      }
    }
  };
  for (const [x, y, h, width, offsets] of [
    [x0, y0, h0, 2.4, [-1.05, 0]],
    [x1, y1, h1, 2.5, [-0.58, 0.58]],
    [x2, y2, h2, 2.5, [-0.58, 0.58]],
  ] as [number, number, number, number, number[]][]) {
    for (const offset of offsets)
      for (const side of [-0.47, 0.47]) {
        const p = local(x, y, h, offset, width * side, 0);
        wheel(p[0], p[1], h);
      }
  }
  for (const side of [-1.2, 1.2]) {
    const p = local(x0, y0, h0, 3.5, side, 0);
    wheel(p[0], p[1], h0 + state.phi);
  }
  box(x0, y0, h1, -2.4, 4.1, 2.5, 1.02, 3.5, [73, 126, 164], true, 0.8);
  box(x2, y2, h2, 3, 7.7, 2.5, 1.02, 3.5, [72, 143, 125], true);
  for (const [x, y, h] of [
    [x0, y0, h0],
    [x1, y1, h1],
  ])
    box(x, y, h, 0, 0.85, 0.9, 0.9, 1.05, [132, 144, 149]);
  box(x0, y0, h0, 1.65, 1.95, 2.4, 0.85, 2.95, [166, 40, 42]);
  box(x0, y0, h0, 1.45, 1.25, 1.824, 2.95, 3.12, [195, 56, 53]);
  const cp = (a: number, b: number, z: number) => local(x0, y0, h0, a, b, z);
  face(
    [cp(2.62, -1.04, 2.85), cp(2.62, 1.04, 2.85), cp(2.95, 1, 1.65), cp(2.95, -1, 1.65)],
    [34, 64, 79],
  );
  face([cp(2.64, 0, 2.8), cp(2.93, 0, 1.7)], [14, 27, 35], 0.065);
  for (const side of [-1, 1]) {
    if ((-Math.sin(h0) * 0.1 - Math.cos(h0) * 0.3) * side > 0) {
      face(
        [
          cp(1.7, side * 1.2, 2.7),
          cp(2.55, side * 1.2, 2.7),
          cp(2.55, side * 1.2, 1.85),
          cp(1.7, side * 1.2, 1.85),
        ],
        [28, 52, 65],
      );
      face([cp(1.86, side * 1.2, 1.62), cp(2.15, side * 1.2, 1.62)], [213, 222, 215], 0.075);
    }
    const tank = cp(0.9, side * 1.05, 0.3);
    box(tank[0], tank[1], h0, 0, 1.45, 0.35, 0.35, 0.85, [153, 166, 169]);
    const mirror = cp(2.6, side * 1.45, 2);
    box(mirror[0], mirror[1], h0, 0, 0.38, 0.18, 1.9, 2.45, [67, 80, 86]);
  }
  box(x0, y0, h0, 3.65, 1.7, 1.824, 0.8, 1.65, [155, 34, 37]);
  box(x0, y0, h0, 4.5, 0.16, 2.256, 0.55, 0.86, [184, 196, 196]);
  face(
    [cp(4.51, -0.57, 0.9), cp(4.51, 0.57, 0.9), cp(4.51, 0.57, 1.55), cp(4.51, -0.57, 1.55)],
    [31, 42, 50],
  );
  for (const z of [1.03, 1.2, 1.37])
    face([cp(4.52, -0.5, z), cp(4.52, 0.5, z)], [124, 143, 149], 0.07);
  for (const side of [-0.78, 0.78])
    face(
      [
        cp(4.53, side - 0.17, 0.96),
        cp(4.53, side + 0.17, 0.96),
        cp(4.53, side + 0.17, 1.25),
        cp(4.53, side - 0.17, 1.25),
      ],
      [255, 242, 204],
    );
  const exhaust = cp(0.5, -0.94, 0);
  box(exhaust[0], exhaust[1], h0, 0, 0.18, 0.18, 0.85, 3.35, [157, 170, 176]);
  box(x2, y2, h2, -1.08, 0.18, 2.5, 0.42, 0.85, [78, 91, 96]);
  for (const side of [-1.08, -0.67, 0.67, 1.08]) {
    const rp = (b: number, z: number) => local(x2, y2, h2, -1.18, b, z);
    face(
      [rp(side - 0.15, 0.53), rp(side + 0.15, 0.53), rp(side + 0.15, 0.77), rp(side - 0.15, 0.77)],
      Math.abs(side) > 1 ? [242, 49, 49] : [238, 232, 206],
    );
  }
  for (const f of faces.sort((a, b) => a.depth - b.depth)) {
    const points = f.points.map(project);
    if (f.width)
      for (let i = 1; i < points.length; i++) mesh.line(points[i - 1], points[i], f.width, f.color);
    else mesh.polygon(points, f.color);
  }
}
export function ghost(mesh: Mesh, state: State, alpha: number) {
  const [[x0, y0, h0], [x1, y1, h1], [x2, y2, h2]] = axles(state);
  for (const [x, y, h, offset, length, width] of [
    [x0, y0, h0, 2, 5, 2.4],
    [x1, y1, h1, 2.25, 7.5, 2.5],
    [x2, y2, h2, 2.25, 7.5, 2.5],
  ]) {
    const points = [
      [-1, -1],
      [1, -1],
      [1, 1],
      [-1, 1],
    ].map(([a, b]) => project(local(x, y, h, offset + (a * length) / 2, (b * width) / 2, 0)));
    for (let i = 0; i < 4; i++)
      mesh.line(points[i], points[(i + 1) % 4], 0.09, [255, 255, 255, alpha]);
  }
}
