/** The desktop's on-axle double-trailer model, in metres, radians and seconds. */
export interface State {
  x0: number;
  y0: number;
  theta0: number;
  beta1: number;
  beta2: number;
  phi: number;
}
export type Point = [number, number];
export const radians = (degrees: number) => (degrees * Math.PI) / 180;
export const clamp = (x: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, x));
export const wrap = (x: number) => Math.atan2(Math.sin(x), Math.cos(x));
export const MAX_STEER = radians(35),
  CRITICAL_HITCH = radians(70),
  DT = 0.02;
export function axles(s: State): [number, number, number][] {
  const t1 = s.theta0 - s.beta1,
    t2 = t1 - s.beta2;
  const x1 = s.x0 - 6 * Math.cos(t1),
    y1 = s.y0 - 6 * Math.sin(t1);
  return [
    [s.x0, s.y0, s.theta0],
    [x1, y1, t1],
    [x1 - 6 * Math.cos(t2), y1 - 6 * Math.sin(t2), t2],
  ];
}
export function step(state: State, speed: number, target: number, dt = DT): State {
  const phi = clamp(
    state.phi + clamp(target - state.phi, -radians(45) * dt, radians(45) * dt),
    -MAX_STEER,
    MAX_STEER,
  );
  const y = [state.x0, state.y0, state.theta0, state.beta1, state.beta2];
  const derivative = (v: number[]) => {
    const [, , theta, b1, b2] = v;
    const d0 = (speed / 3.5) * Math.tan(phi),
      d1 = (speed / 6) * Math.sin(b1),
      d2 = ((speed * Math.cos(b1)) / 6) * Math.sin(b2);
    return [speed * Math.cos(theta), speed * Math.sin(theta), d0, d0 - d1, d1 - d2];
  };
  const offset = (k: number[], scale: number) => y.map((v, i) => v + scale * k[i]);
  const k1 = derivative(y),
    k2 = derivative(offset(k1, dt / 2)),
    k3 = derivative(offset(k2, dt / 2)),
    k4 = derivative(offset(k3, dt));
  const next = y.map((v, i) => v + (dt / 6) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]));
  return {
    x0: next[0],
    y0: next[1],
    theta0: wrap(next[2]),
    beta1: wrap(next[3]),
    beta2: wrap(next[4]),
    phi,
  };
}
export function initialState(start: Point, heading: number): State {
  const theta0 = heading - Math.PI;
  return {
    x0: start[0] + 12 * Math.cos(theta0),
    y0: start[1] + 12 * Math.sin(theta0),
    theta0,
    beta1: 0,
    beta2: 0,
    phi: 0,
  };
}
