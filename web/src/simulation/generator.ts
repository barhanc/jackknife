/** Procedural families and lot fitting from Python's path_generator and game sampler. */
import { clamp, type Point } from './physics';
import { drawnRoute, gaussian, pathFromPoints, roundEven, type Route } from './path';
export const PATH_FAMILIES = [
  'random_fourier',
  'yard_grand_tour',
  'yard_serpentine_slalom',
  'yard_double_figure8',
  'yard_perimeter_clover',
  'self_intersecting_figure8',
  'loop_the_loop',
  'sharp_switchbacks',
  'cloverleaf_intersect',
  'serpentine_knot',
  'dock_with_360_turnaround',
  'high_curvature_chicane',
  'multi_scale_s',
  'segmented_dock',
  'continuous_arc',
  'figure8_inflection',
  'compound_kink',
  'piecewise_clothoid',
  'rapid_chicane',
  'z_track',
  'jagged_corners',
] as const;
export type Family = (typeof PATH_FAMILIES)[number];
export interface RandomSource {
  uniform(low: number, high: number): number;
  integer(low: number, high: number): number;
  sign(): number;
  random(): number;
  sample(population: number, count: number): number[];
}
export function randomSource(random = Math.random): RandomSource {
  return {
    random,
    uniform: (a, b) => a + (b - a) * random(),
    integer: (a, b) => a + Math.floor((b - a) * random()),
    sign: () => (random() < 0.5 ? -1 : 1),
    sample(n, count) {
      const pool = Array.from({ length: n }, (_, i) => i);
      for (let i = 0; i < count; i++) {
        const j = i + Math.floor(random() * (n - i));
        [pool[i], pool[j]] = [pool[j], pool[i]];
      }
      return pool.slice(0, count);
    },
  };
}
const linspace = (a: number, b: number, n: number) =>
  Array.from({ length: n }, (_, i) => (n === 1 ? a : a + ((b - a) * i) / (n - 1)));
const zeros = (n: number) => Array(Math.max(0, n)).fill(0) as number[];
const full = (n: number, value: number) => Array(Math.max(0, n)).fill(value) as number[];
const cumulative = (values: number[]) => {
  let sum = 0;
  return values.map((v) => (sum += v));
};
function integrate(curvatures: number[], ds: number): Point[] {
  const headings = cumulative(curvatures.map((k) => k * ds)),
    x = cumulative(headings.map((h) => Math.cos(h) * ds)),
    y = cumulative(headings.map((h) => Math.sin(h) * ds));
  return x.map((v, i) => [v, y[i]]);
}
const smooth = (points: Point[], radius: number) =>
  drawnRoute(points, { leadIn: 0, minRadius: radius }).points;
export function generateFamily(
  family: Family,
  length: number,
  rng: RandomSource = randomSource(),
  rotation = true,
  maxCurvature = 0.11,
): Route {
  const ds = 0.8,
    n = Math.trunc(length / ds) + 1,
    stations = linspace(0, length, n),
    pi = Math.PI;
  const u = (a: number, b: number) => rng.uniform(a, b),
    integer = (a: number, b: number) => rng.integer(a, b),
    sign = () => rng.sign();
  let points: Point[] = [];
  const integrateLimited = (k: number[]) =>
    integrate(
      k.map((k) => clamp(k, -maxCurvature, maxCurvature)),
      ds,
    );
  switch (family) {
    case 'random_fourier': {
      const count = integer(2, 6),
        k = zeros(n);
      for (let h = 1; h <= count; h++) {
        const frequency = u(0.015, 0.075),
          amplitude = u(0.015, 0.045) / h,
          phase = u(0, 2 * pi);
        stations.forEach((s, i) => (k[i] += amplitude * Math.sin(frequency * s + phase)));
      }
      points = integrateLimited(k);
      break;
    }
    case 'continuous_arc': {
      const radius = u(12, 48),
        k = sign() / radius,
        lead = roundEven(u(8, 18) / ds),
        ramp = roundEven(u(10, 20) / ds),
        out = roundEven(u(8, 18) / ds),
        hold = Math.max(10, n - lead - out - 2 * ramp);
      const kappa = [
        ...zeros(lead),
        ...linspace(0, k, ramp),
        ...full(hold, k),
        ...linspace(k, 0, ramp),
        ...zeros(out),
        ...zeros(n),
      ];
      points = integrate(kappa.slice(0, n), ds);
      break;
    }
    case 'figure8_inflection': {
      const k1 = 1 / u(18, 44),
        k2 = -1 / u(18, 44),
        direction = sign(),
        lead = roundEven(u(10, 20) / ds),
        cross = roundEven(u(20, 45) / ds),
        ramp = roundEven(u(12, 20) / ds),
        rem = Math.max(16, n - lead - cross - 4 * ramp - roundEven(10 / ds)),
        hold = Math.max(8, Math.floor(rem / 2));
      const k = [
        ...zeros(lead),
        ...linspace(0, direction * k1, ramp),
        ...full(hold, direction * k1),
        ...linspace(direction * k1, 0, ramp),
        ...zeros(cross),
        ...linspace(0, direction * k2, ramp),
        ...full(hold, direction * k2),
        ...linspace(direction * k2, 0, ramp),
        ...zeros(Math.max(4, n - (lead + 4 * ramp + 2 * hold + cross))),
        ...zeros(n),
      ];
      points = integrate(k.slice(0, n), ds);
      break;
    }
    case 'multi_scale_s': {
      const wavelength = u(28, 130),
        amplitude = u(3.5, 16),
        phase = u(0, 2 * pi);
      points = integrateLimited(
        stations.map(
          (s) =>
            amplitude * ((2 * pi) / wavelength) ** 2 * Math.sin((2 * pi * s) / wavelength + phase),
        ),
      );
      break;
    }
    case 'jagged_corners': {
      const legs = integer(3, 8),
        raw: Point[] = [[0, 0]];
      let x = 0,
        y = 0,
        h = 0;
      for (let i = 0; i < legs; i++) {
        h += u((-65 * pi) / 180, (65 * pi) / 180);
        x += (length / legs) * u(0.8, 1.2) * Math.cos(h);
        y += (length / legs) * u(0.8, 1.2) * Math.sin(h);
        raw.push([x, y]);
      }
      points = smooth(raw, 10);
      break;
    }
    case 'compound_kink': {
      const offset = sign() * u(3, 10),
        width = u(14, 26),
        start = length * u(0.2, 0.35);
      points = smooth(
        [
          [0, 0],
          [start, 0],
          [start + width, offset],
          [length * u(0.7, 0.85), offset],
          [length, offset],
        ],
        10,
      );
      break;
    }
    case 'z_track': {
      const span = u(48, 64),
        slope = u(0.35, 0.55),
        offset = u(10, 18);
      points = smooth(
        [
          [span, span * slope + offset],
          [span * 0.65, span * 0.65 * slope + offset],
          [span * 0.35, span * 0.35 * slope + offset],
          [span * 0.1, span * 0.1 * slope + offset * 0.6],
          [-span * 0.1, -span * 0.1 * slope],
          [-span * 0.35, -span * 0.35 * slope],
          [-span * 0.65, -span * 0.65 * slope],
          [-span, -span * slope],
        ],
        12.5,
      );
      break;
    }
    case 'segmented_dock': {
      const k = zeros(n),
        start = Math.trunc(n * u(0.18, 0.35)),
        turnLength = Math.trunc(u(16, 36) / ds),
        end = Math.min(n - 8, start + turnLength),
        direction = sign(),
        turn = direction * u(0.035, maxCurvature);
      linspace(0, 1, Math.max(2, end - start)).forEach(
        (t, i) => (k[start + i] = turn * Math.sin(pi * t) ** 2),
      );
      if (rng.random() < 0.6) {
        const a = Math.min(n - 15, end + Math.trunc(u(10, 20) / ds)),
          length = Math.trunc(u(14, 26) / ds),
          b = Math.min(n - 4, a + length),
          magnitude = -turn * u(0.5, 0.95);
        linspace(0, 1, Math.max(2, b - a)).forEach(
          (t, i) => (k[a + i] = magnitude * Math.sin(pi * t) ** 2),
        );
      }
      points = integrate(k, ds);
      break;
    }
    case 'rapid_chicane': {
      const center = Math.trunc(n * u(0.35, 0.6)),
        width = Math.trunc(n * u(0.08, 0.16)),
        direction = sign(),
        magnitude = u(0.045, maxCurvature),
        k = zeros(n);
      k.fill(direction * magnitude, center - width, center);
      k.fill(-direction * magnitude, center, center + width);
      points = integrate(k, ds);
      break;
    }
    case 'piecewise_clothoid': {
      const k = zeros(n),
        count = integer(3, 8),
        splits = [
          0,
          ...rng
            .sample(n - 2, count)
            .sort((a, b) => a - b)
            .map((x) => x + 1),
          n,
        ];
      for (let i = 0; i < splits.length - 1; i++)
        k.fill(i % 2 ? u(-maxCurvature, maxCurvature) : 0, splits[i], splits[i + 1]);
      points = integrate(gaussian(k, u(1.5, 3), true), ds);
      break;
    }
    case 'self_intersecting_figure8': {
      const a = u(28, 56),
        b = u(18, 38),
        jitter = u(-0.15, 0.15),
        t = linspace(-pi * u(0.92, 0.98), pi * u(0.92, 0.98), n),
        loop: Point[] = t.map((t) => [a * Math.sin(t), b * Math.sin(2 * t + jitter) * 0.5]);
      const first = loop[0],
        last = loop.at(-1)!,
        h0 = Math.atan2(loop[1][1] - first[1], loop[1][0] - first[0]),
        h1 = Math.atan2(last[1] - loop.at(-2)![1], last[0] - loop.at(-2)![0]);
      const lead = linspace(-u(14, 24), 0, Math.max(5, Math.trunc(18 / ds))),
        out = linspace(0, u(14, 24), Math.max(5, Math.trunc(18 / ds)));
      points = [
        ...lead
          .slice(0, -1)
          .map((s): Point => [first[0] + s * Math.cos(h0), first[1] + s * Math.sin(h0)]),
        ...loop,
        ...out.slice(1).map((s): Point => [last[0] + s * Math.cos(h1), last[1] + s * Math.sin(h1)]),
      ];
      break;
    }
    case 'loop_the_loop':
    case 'dock_with_360_turnaround': {
      const radius = family === 'loop_the_loop' ? u(11, 18.5) : u(11.5, 17.5);
      let direction: number, lead: number[], angles: number[];
      if (family === 'loop_the_loop') {
        direction = sign();
        const leadLength = u(16, 26);
        lead = linspace(-leadLength, 0, Math.max(5, Math.trunc(leadLength / ds)));
        const span = 2 * pi + u(0.4, 1.6);
        angles = linspace(0, span, Math.max(15, Math.trunc((span * radius) / ds)));
      } else {
        const span = 2 * pi + u(0.3, pi * 0.75);
        angles = linspace(0, span, Math.max(20, Math.trunc((span * radius) / ds)));
        direction = sign();
        lead = linspace(-u(14, 22), 0, Math.max(5, Math.trunc(18 / ds)));
      }
      const loop: Point[] = angles.map((t) => [
          radius * Math.sin(t),
          direction * radius * (1 - Math.cos(t)),
        ]),
        end = loop.at(-1)!,
        angle = angles.at(-1)!,
        exitLength = u(16, 26),
        out = linspace(0, exitLength, Math.max(5, Math.trunc(exitLength / ds)));
      points = [
        ...lead.slice(0, -1).map((s): Point => [s, 0]),
        ...loop,
        ...out
          .slice(1)
          .map((s): Point => [
            end[0] + s * Math.cos(angle),
            end[1] + s * direction * Math.sin(angle),
          ]),
      ];
      break;
    }
    case 'sharp_switchbacks': {
      const count = integer(2, 5),
        k = zeros(Math.max(4, Math.trunc(14 / ds)));
      let direction = sign();
      for (let i = 0; i < count; i++) {
        const radius = u(10.2, 14.5),
          arc = pi * radius * u(0.7, 0.98);
        k.push(
          ...full(Math.max(4, Math.trunc(arc / ds)), direction / radius),
          ...zeros(Math.max(4, Math.trunc(u(12, 24) / ds))),
        );
        direction = -direction;
      }
      points = integrate(k, ds);
      break;
    }
    case 'cloverleaf_intersect': {
      const a = u(34, 52),
        b = u(18, 32),
        c = u(6, 14),
        frequency = u(0.85, 1.15),
        t = linspace(-pi * u(0.88, 0.96), pi * u(0.88, 0.96), n);
      points = t.map((t) => [
        a * Math.sin(frequency * t),
        b * Math.sin(2 * frequency * t) + c * Math.sin(frequency * t),
      ]);
      break;
    }
    case 'serpentine_knot': {
      const a1 = u(28, 44),
        a2 = u(8, 16),
        b = u(18, 30),
        w1 = u(1.2, 1.7),
        w2 = u(0.6, 1),
        w3 = u(1.8, 2.4),
        phase = u(0, pi);
      points = linspace(-2.2, 2.2, n).map((t) => [
        a1 * Math.sin(t * w1) + a2 * Math.cos(t * w2 + phase),
        b * Math.cos(t * w3),
      ]);
      break;
    }
    case 'high_curvature_chicane': {
      const radius = u(10, 13.5),
        direction = sign(),
        width = Math.max(5, Math.trunc(u(10, 16) / ds)),
        k = zeros(n),
        mid = Math.floor(n / 2);
      k.fill(direction / radius, mid - width, mid);
      k.fill(-direction / radius, mid, mid + width);
      points = integrate(k, ds);
      break;
    }
    case 'yard_grand_tour': {
      const sectors = integer(8, 14);
      let angles = Array.from({ length: sectors }, (_, i) => (i * 2 * pi) / sectors);
      if (rng.random() < 0.5) angles.reverse();
      const angle = u(0, 2 * pi);
      angles = angles.map((a) => (a + angle) % (2 * pi)).sort((a, b) => a - b);
      if (rng.random() >= 0.5) angles.reverse();
      const raw: Point[] = angles.map((a) => {
        const fraction = u(0.4, 1),
          rx = fraction * u(48, 62),
          ry = fraction * u(22, 32);
        return [rx * Math.cos(a), ry * Math.sin(a)];
      });
      const last = raw.at(-1)!;
      raw.push([last[0] * 0.4 + u(-10, 10), last[1] * 0.4 + u(-10, 10)], [u(-20, 20), 0]);
      points = smooth(raw, 11);
      break;
    }
    case 'yard_serpentine_slalom': {
      const direction = sign(),
        gates = integer(4, 9),
        start = u(-64, -56),
        end = u(56, 64),
        xs = linspace(start, end, gates).map((x) => x + u(-3.5, 3.5));
      const raw: Point[] = xs.map((x, i) => [x, direction * (i % 2 ? -1 : 1) * u(22, 32)]);
      raw.push([end + u(4, 8), 0]);
      points = smooth(raw, 11);
      break;
    }
    case 'yard_double_figure8': {
      const a1 = u(48, 62),
        a2 = u(4, 10),
        b = u(24, 32);
      points = smooth(
        linspace(-pi, pi, Math.max(200, Math.trunc(length / ds))).map((t): Point => [
          a1 * Math.sin(t) + a2 * Math.sin(3 * t),
          b * Math.sin(2 * t),
        ]),
        10.5,
      );
      break;
    }
    case 'yard_perimeter_clover': {
      const a = u(50, 62),
        b1 = u(24, 32),
        b2 = u(8, 14);
      points = smooth(
        linspace(-pi, pi, Math.max(220, Math.trunc(length / ds))).map((t): Point => [
          a * Math.sin(t),
          b1 * Math.sin(2 * t) + b2 * Math.sin(t),
        ]),
        10.5,
      );
      break;
    }
  }
  if (rotation) {
    const angle = u(0, 2 * pi),
      flip = sign(),
      c = Math.cos(angle),
      s = Math.sin(angle);
    points = points.map(([x, y]) => [c * x - s * flip * y, s * x + c * flip * y]);
  }
  return pathFromPoints(points);
}
const grand: readonly string[] = [
  'yard_grand_tour',
  'yard_double_figure8',
  'yard_perimeter_clover',
];
const long: readonly string[] = [
  'yard_serpentine_slalom',
  'self_intersecting_figure8',
  'cloverleaf_intersect',
  'serpentine_knot',
  'random_fourier',
];
const loops: readonly string[] = ['loop_the_loop', 'dock_with_360_turnaround', 'z_track'];
export function sampleRoute(rng: RandomSource = randomSource()): Route & { family: string } {
  for (let attempt = 0; attempt < 16; attempt++) {
    const family = PATH_FAMILIES[rng.integer(0, PATH_FAMILIES.length)],
      length = grand.includes(family)
        ? rng.uniform(220, 320)
        : long.includes(family)
          ? rng.uniform(140, 240)
          : loops.includes(family)
            ? rng.uniform(105, 150)
            : rng.uniform(70, 110);
    const raw = generateFamily(family, length, rng),
      xs = raw.points.map((p) => p[0]),
      ys = raw.points.map((p) => p[1]);
    let points = raw.points;
    if (Math.max(...ys) - Math.min(...ys) > Math.max(...xs) - Math.min(...xs))
      points = points.map(([x, y]) => [-y, x]);
    const flipX = rng.random() < 0.5 ? -1 : 1,
      flipY = rng.random() < 0.5 ? -1 : 1;
    points = points.map(([x, y]) => [x * flipX, y * flipY]);
    const minX = Math.min(...points.map((p) => p[0])),
      maxX = Math.max(...points.map((p) => p[0])),
      minY = Math.min(...points.map((p) => p[1])),
      maxY = Math.max(...points.map((p) => p[1]));
    const maxScale = [...grand, 'yard_serpentine_slalom'].includes(family) ? 1.35 : 1.15,
      scale = Math.min(
        maxScale,
        124 / Math.max(0.001, maxX - minX),
        68 / Math.max(0.001, maxY - minY),
      );
    points = points.map(([x, y]) => [
      (x - (minX + maxX) / 2) * scale,
      (y - (minY + maxY) / 2) * scale,
    ]);
    const heading = Math.atan2(points[1][1] - points[0][1], points[1][0] - points[0][0]),
      cab: Point = [
        points[0][0] + 11.5 * Math.cos(heading - Math.PI),
        points[0][1] + 11.5 * Math.sin(heading - Math.PI),
      ];
    if ([...points, cab].every(([x, y]) => x >= -70 && x <= 70 && y >= -40 && y <= 40))
      return { ...pathFromPoints(points, false), family };
  }
  const raw = generateFamily('multi_scale_s', 72, rng, true, 0.08),
    xs = raw.points.map((p) => p[0]),
    ys = raw.points.map((p) => p[1]),
    cx = (Math.min(...xs) + Math.max(...xs)) / 2,
    cy = (Math.min(...ys) + Math.max(...ys)) / 2;
  return {
    ...pathFromPoints(raw.points.map(([x, y]) => [x - cx, y - cy])),
    family: 'multi_scale_s',
  };
}
