/** Arc-length routes and rear-trailer policy features, matching the Python game. */
import { clamp, type Point } from './physics';
export interface Route {
  points: Point[];
  headings: number[];
  curvatures: number[];
  length: number;
}
const distance = (a: Point, b: Point) => Math.hypot(a[0] - b[0], a[1] - b[1]);
function arcs(points: Point[]) {
  const s = [0];
  for (let i = 1; i < points.length; i++)
    s.push(s[i - 1] + Math.max(1e-6, distance(points[i], points[i - 1])));
  return s;
}
function gradient(values: number[], s: number[]) {
  return values.map((v, i) => {
    if (i === 0) return (values[1] - v) / (s[1] - s[0]);
    if (i === values.length - 1) return (v - values[i - 1]) / (s[i] - s[i - 1]);
    const a = s[i] - s[i - 1],
      b = s[i + 1] - s[i];
    return (
      (-b / (a * (a + b))) * values[i - 1] +
      ((b - a) / (a * b)) * v +
      (a / (b * (a + b))) * values[i + 1]
    );
  });
}
function geometry(points: Point[], s = arcs(points)) {
  const dx = gradient(
      points.map((p) => p[0]),
      s,
    ),
    dy = gradient(
      points.map((p) => p[1]),
      s,
    );
  const headings = dx.map((x, i) => Math.atan2(dy[i], x));
  const unwrapped = [headings[0]];
  for (let i = 1; i < headings.length; i++) {
    let delta = headings[i] - headings[i - 1];
    delta -= Math.round(delta / (2 * Math.PI)) * 2 * Math.PI;
    unwrapped.push(unwrapped[i - 1] + delta);
  }
  return { headings, curvatures: gradient(unwrapped, s) };
}
function resample(points: Point[], ds = 0.8) {
  const s = arcs(points),
    length = s.at(-1)!;
  const count = Math.max(2, Math.ceil(length / ds) + 1);
  let j = 0;
  const samples = Array.from({ length: count }, (_, i) => (length * i) / (count - 1));
  const result: Point[] = samples.map((t) => {
    while (j < s.length - 2 && s[j + 1] < t) j++;
    const f = (t - s[j]) / (s[j + 1] - s[j]);
    return [
      points[j][0] + f * (points[j + 1][0] - points[j][0]),
      points[j][1] + f * (points[j + 1][1] - points[j][1]),
    ];
  });
  return { points: result, samples, length };
}
export function gaussian(values: number[], sigma: number, reflect = false) {
  const radius = Math.round(4 * sigma);
  const weights = Array.from({ length: radius * 2 + 1 }, (_, i) =>
    Math.exp(-0.5 * ((i - radius) / sigma) ** 2),
  );
  const sum = weights.reduce((a, b) => a + b, 0);
  const at = (index: number) => {
    if (!reflect) return values[clamp(index, 0, values.length - 1)];
    const period = values.length * 2;
    const wrapped = ((index % period) + period) % period;
    return values[wrapped < values.length ? wrapped : period - wrapped - 1];
  };
  return values.map(
    (_, i) => weights.reduce((total, w, k) => total + w * at(i + k - radius), 0) / sum,
  );
}
function blur(points: Point[], sigma: number): Point[] {
  const x = gaussian(
      points.map((p) => p[0]),
      sigma,
    ),
    y = gaussian(
      points.map((p) => p[1]),
      sigma,
    );
  return x.map((v, i) => [v, y[i]]);
}
export function drawnRoute(raw: Point[], { leadIn = 12, minRadius = 12.5 } = {}): Route {
  const points = raw.filter(
    (p, i) => i === 0 || i === raw.length - 1 || distance(p, raw[i - 1]) > 0.4,
  );
  if (points.length < 2 || arcs(points).at(-1)! < 6)
    throw new Error('Draw a route at least 6 metres long.');
  const sampled = resample(points),
    { samples: s, length } = sampled;
  const start = points[0],
    end = points.at(-1)!;
  let curve = blur(sampled.points, 1.2);
  curve[0] = start;
  curve[curve.length - 1] = end;
  let pinned = 0;
  let straight = curve.slice();
  if (leadIn > 0) {
    const lead = Math.min(leadIn, length / 3);
    pinned = Math.min(
      curve.length - 2,
      Math.max(
        1,
        s.findIndex((t) => t >= lead),
      ),
    );
    let delta: Point = [curve[pinned][0] - start[0], curve[pinned][1] - start[1]];
    if (Math.hypot(...delta) < 1e-6) {
      const farthest = curve
        .slice(0, pinned + 1)
        .sort((a, b) => distance(b, start) - distance(a, start))[0];
      delta = [farthest[0] - start[0], farthest[1] - start[1]];
    }
    straight = s.map((t) => [
      start[0] + (t / s[pinned]) * delta[0],
      start[1] + (t / s[pinned]) * delta[1],
    ]);
    const min = [Math.min(...points.map((p) => p[0])), Math.min(...points.map((p) => p[1]))];
    const max = [Math.max(...points.map((p) => p[0])), Math.max(...points.map((p) => p[1]))];
    curve = curve.map((p, i) => {
      let f = clamp((s[i] - s[pinned]) / Math.max(8, Math.min(12, length - lead)), 0, 1);
      f = f * f * (3 - 2 * f);
      return i <= pinned
        ? straight[i]
        : [
            clamp(straight[i][0] * (1 - f) + p[0] * f, min[0], max[0]),
            clamp(straight[i][1] * (1 - f) + p[1] * f, min[1], max[1]),
          ];
    });
    curve[curve.length - 1] = end;
  }
  const targetCurvature = 0.95 * Math.min(0.12, 1 / Math.max(5, minRadius));
  for (let iteration = 0; iteration < 400; iteration++) {
    const tight = geometry(curve).curvatures.map((k) => (Math.abs(k) > targetCurvature ? 1 : 0));
    if (!tight.some(Boolean)) break;
    const weights = gaussian(tight, 2),
      candidate = blur(curve, 1.2);
    curve = curve.map((p, i) =>
      i <= pinned
        ? straight[i]
        : i === curve.length - 1
          ? end
          : [
              p[0] + Math.min(1, 2 * weights[i]) * (candidate[i][0] - p[0]),
              p[1] + Math.min(1, 2 * weights[i]) * (candidate[i][1] - p[1]),
            ],
    );
  }
  const result = resample(curve);
  const shape = geometry(result.points, result.samples);
  return {
    points: result.points,
    headings: shape.headings,
    curvatures: shape.curvatures.map((k) => clamp(k, -0.12, 0.12)),
    length: result.length,
  };
}
// Python rounds ties to even; preserving it avoids progress differences at half-stations.
export const roundEven = (v: number) =>
  Math.abs(v % 1) === 0.5 ? 2 * Math.round(v / 2) : Math.round(v);
export function progression(route: Route, x: number, y: number, index: number) {
  const p = route.points[index],
    h = route.headings[index];
  const target = index + roundEven(((x - p[0]) * Math.cos(h) + (y - p[1]) * Math.sin(h)) / 0.8);
  const next = clamp(target, 0, route.points.length - 1),
    q = route.points[next],
    th = route.headings[next];
  const lateral = Math.abs(-(x - q[0]) * Math.sin(th) + (y - q[1]) * Math.cos(th));
  const overshoot =
    target < 0
      ? Math.max(
          0,
          -(
            (x - route.points[0][0]) * Math.cos(route.headings[0]) +
            (y - route.points[0][1]) * Math.sin(route.headings[0])
          ) - 1.5,
        ) * 1.5
      : 0;
  return { index: next, error: Math.hypot(lateral, overshoot) };
}
export function tokensFor(route: Route, x: number, y: number, theta: number, index: number) {
  const tokens = new Float32Array(448),
    c = Math.cos(theta),
    s = Math.sin(theta);
  for (let i = 0; i < 64; i++) {
    const j = Math.min(index + i, route.points.length - 1),
      p = route.points[j];
    const dx = p[0] - x,
      dy = p[1] - y,
      h = route.headings[j] - theta;
    tokens.set(
      [c * dx + s * dy, -s * dx + c * dy, Math.cos(h), Math.sin(h), route.curvatures[j], 3.5, 3.5],
      i * 7,
    );
  }
  return tokens;
}

/** Equivalent to Python Path construction, including its optional Gaussian pass. */
export function pathFromPoints(raw: Point[], constrain = true, sigma = 1.5): Route {
  let points = raw.filter((p, i) => i === 0 || distance(p, raw[i - 1]) > 0.001);
  if (points.length < 2) points = [raw[0], [raw[0][0] + 1, raw[0][1]]];
  const result = resample(points);
  const smoothed =
    constrain && sigma > 0 && result.points.length > 6 ? blur(result.points, sigma) : result.points;
  const shape = geometry(smoothed, result.samples),
    limit = constrain ? 0.12 : 0.35;
  return {
    points: smoothed,
    headings: shape.headings,
    curvatures: shape.curvatures.map((k) => clamp(k, -limit, limit)),
    length: result.length,
  };
}
