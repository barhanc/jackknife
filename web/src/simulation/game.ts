/** Fixed-step game state. Policy readbacks apply only to the generation that requested them. */
import type { Policy } from '../inference/policy';
import {
  axles,
  clamp,
  CRITICAL_HITCH,
  DT,
  initialState,
  MAX_STEER,
  radians,
  step,
  type Point,
} from './physics';
import { progression, tokensFor, type Route } from './path';
export class Game {
  state;
  mode: 'MANUAL' | 'AUTONOMOUS' = 'AUTONOMOUS';
  speedMultiplier = 16;
  speed = 0;
  targetSteer = 0;
  index = 0;
  error = 0;
  status: '' | 'JACKKNIFED' | 'OFF TRACK' | 'DOCK SECURED' = '';
  paused = false;
  drawing = false;
  rays = false;
  diagnostics = true;
  probabilities = Array(15).fill(1 / 15) as number[];
  attention = Array(64).fill(1 / 64) as number[];
  inferenceMs = 0;
  private inferenceStatsTime = 0;
  simulatedSeconds = 0;
  keys = new Set<string>();
  accumulator = 0;
  private generation = 0;
  private cadence = 0;
  private steeringHold = 0;
  private advancing = false;
  failure = '';
  constructor(
    public route: Route,
    readonly policy: Policy,
  ) {
    this.state = initialState(route.points[0], route.headings[0]);
  }
  reset(route = this.route) {
    this.generation++;
    this.route = route;
    this.state = initialState(route.points[0], route.headings[0]);
    this.speed =
      this.targetSteer =
      this.index =
      this.error =
      this.accumulator =
      this.cadence =
      this.steeringHold =
      this.simulatedSeconds =
        0;
    this.status = '';
    this.failure = '';
    this.inferenceMs = 0;
    this.inferenceStatsTime = 0;
    this.attention.fill(1 / 64);
    this.probabilities.fill(1 / 15);
  }
  switchMode() {
    this.generation++;
    this.mode = this.mode === 'MANUAL' ? 'AUTONOMOUS' : 'MANUAL';
    if (this.mode === 'AUTONOMOUS') this.speedMultiplier = 16;
    else this.speed = 0;
    this.cadence = 0;
    this.accumulator = 0;
    this.keys.clear();
  }
  setPaused(paused: boolean) {
    this.paused = paused;
    this.generation++;
    this.accumulator = 0;
  }
  setDrawing(drawing: boolean) {
    this.drawing = drawing;
    this.generation++;
    this.accumulator = 0;
    this.keys.clear();
  }
  private controls() {
    if (this.mode !== 'MANUAL') {
      this.speed = -1.5;
      return;
    }
    const left = this.keys.has('a') || this.keys.has('arrowleft'),
      right = this.keys.has('d') || this.keys.has('arrowright');
    if (left || right) {
      this.steeringHold += DT;
      const rate = radians(22) * Math.min(1.35, 0.55 + this.steeringHold * 0.55);
      this.targetSteer = clamp(
        this.targetSteer + (left ? 1 : -1) * rate * DT,
        -MAX_STEER,
        MAX_STEER,
      );
    } else {
      this.steeringHold = 0;
      this.targetSteer *= 0.9;
    }
    if (this.keys.has('w') || this.keys.has('arrowup')) this.speed = 2;
    else if (this.keys.has('s') || this.keys.has('arrowdown')) this.speed = -1.5;
    else {
      this.speed *= 0.85;
      if (Math.abs(this.speed) < 0.05) this.speed = 0;
    }
  }
  private checkStatus() {
    if (Math.max(Math.abs(this.state.beta1), Math.abs(this.state.beta2)) >= CRITICAL_HITCH) {
      this.status = 'JACKKNIFED';
      return;
    }
    const [x, y] = axles(this.state)[2];
    const progress = progression(this.route, x, y, this.index);
    this.index = progress.index;
    this.error = progress.error;
    const p = this.route.points[0],
      h = this.route.headings[0];
    const fx = this.state.x0 + 3.5 * Math.cos(this.state.theta0),
      fy = this.state.y0 + 3.5 * Math.sin(this.state.theta0);
    const cabError = Math.abs(-(fx - p[0]) * Math.sin(h) + (fy - p[1]) * Math.cos(h));
    if (this.error > 5.5 || (this.index <= 5 && cabError > 5.5)) {
      this.status = 'OFF TRACK';
      return;
    }
    if (
      this.mode === 'AUTONOMOUS' &&
      this.speed < 0 &&
      this.index >= this.route.points.length - 6
    ) {
      const end = this.route.points.at(-1)!,
        th = this.route.headings.at(-1)!;
      const past = (x - end[0]) * Math.cos(th) + (y - end[1]) * Math.sin(th);
      const lateral = Math.abs(-(x - end[0]) * Math.sin(th) + (y - end[1]) * Math.cos(th));
      if (past >= -2.2 && past <= 6 && lateral <= 4.8) this.status = 'DOCK SECURED';
    }
  }
  async advance(elapsed: number) {
    if (this.paused || this.drawing || this.status || this.failure) {
      this.accumulator = 0;
      return;
    }
    this.accumulator = Math.min(
      this.accumulator + Math.min(elapsed, 0.25) * this.speedMultiplier,
      4,
    );
    if (this.advancing) return;
    this.advancing = true;
    const generation = this.generation;
    try {
      while (this.accumulator + 1e-12 >= DT) {
        this.checkStatus();
        if (this.status) {
          this.speed = 0;
          this.accumulator = 0;
          break;
        }
        this.controls();
        if (this.mode === 'AUTONOMOUS' && this.cadence === 0) {
          const [x, y, theta] = axles(this.state)[2];
          const prediction = await this.policy.predict(
            tokensFor(this.route, x, y, theta, this.index),
            new Float32Array([
              Math.sin(this.state.beta1),
              Math.cos(this.state.beta1),
              Math.sin(this.state.beta2),
              Math.cos(this.state.beta2),
            ]),
          );
          if (generation !== this.generation) return;
          this.targetSteer = prediction.steer;
          this.probabilities = prediction.probabilities;
          this.attention = prediction.attention;
          const now = performance.now();
          const weight = this.inferenceStatsTime
            ? 1 - Math.exp(-(now - this.inferenceStatsTime) / 500)
            : 1;
          this.inferenceMs += weight * (prediction.milliseconds - this.inferenceMs);
          this.inferenceStatsTime = now;
        }
        if (generation !== this.generation) return;
        this.state = step(this.state, this.speed, this.targetSteer);
        this.cadence = (this.cadence + 1) % 4;
        this.accumulator = Math.max(0, this.accumulator - DT);
        this.simulatedSeconds += DT;
      }
    } catch (e) {
      this.failure = e instanceof Error ? e.message : String(e);
      this.setPaused(true);
    } finally {
      this.advancing = false;
    }
  }
  get stability() {
    return clamp(
      1 - Math.max(Math.abs(this.state.beta1), Math.abs(this.state.beta2)) / CRITICAL_HITCH,
      0,
      1,
    );
  }
  get tail(): Point {
    const [x, y] = axles(this.state)[2];
    return [x, y];
  }
}
