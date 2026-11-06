import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { axles, initialState, step, type Point, type State } from '../src/simulation/physics';
import { drawnRoute, progression, tokensFor, type Route } from '../src/simulation/path';
import { Game } from '../src/simulation/game';
import type { Policy, Prediction } from '../src/inference/policy';
const fixtures = JSON.parse(
  readFileSync(new URL('../public/models/parity.json', import.meta.url), 'utf8'),
);
const route: Route = JSON.parse(
  readFileSync(new URL('../public/models/track.json', import.meta.url), 'utf8'),
);

describe('Python/TypeScript agreement', () => {
  it('matches 150 rate-limited RK4 steps', () => {
    let state: State = fixtures.physics.initial;
    for (let i = 0; i < 150; i++) state = step(state, -1.5, -0.18);
    for (const key of Object.keys(state) as (keyof State)[])
      expect(state[key]).toBeCloseTo(fixtures.physics.final[key], 11);
  });
  it('places the rear axle at the route start', () => {
    const tail = axles(initialState(route.points[0], route.headings[0]))[2];
    expect(tail[0]).toBeCloseTo(route.points[0][0], 12);
    expect(tail[1]).toBeCloseTo(route.points[0][1], 12);
  });
  it('matches the policy feature vectors, including padded end stations', () => {
    for (const sample of fixtures.cases) {
      const result = tokensFor(route, sample.pose[0], sample.pose[1], sample.pose[2], sample.index);
      const expected = sample.tokens.flat();
      for (let i = 0; i < result.length; i++) expect(result[i]).toBeCloseTo(expected[i], 5);
    }
  });
  it('matches Python drawn-path smoothing and its straight lead-in', () => {
    const result = drawnRoute(fixtures.drawing.raw);
    expect(result.points.length).toBe(fixtures.drawing.points.length);
    for (let i = 0; i < result.points.length; i++) {
      expect(result.points[i][0]).toBeCloseTo(fixtures.drawing.points[i][0], 6);
      expect(result.points[i][1]).toBeCloseTo(fixtures.drawing.points[i][1], 6);
    }
    expect(result.points[0]).toEqual(fixtures.drawing.raw[0]);
    expect(result.points.at(-1)).toEqual(fixtures.drawing.raw.at(-1));
    for (let i = 1; i < 8; i++) expect(result.headings[i]).toBeCloseTo(result.headings[0], 9);
  });
});
it('keeps progress on its branch at an intersection', () => {
  const points: Point[] = Array.from({ length: 51 }, (_, i) => [(i - 25) * 0.8, 0]);
  const crossing: Route = {
    points: [...points, [0, -1], [0, 0], [0, 1]],
    headings: [...points.map(() => 0), Math.PI / 2, Math.PI / 2, Math.PI / 2],
    curvatures: Array(54).fill(0),
    length: 50,
  };
  expect(progression(crossing, 0.1, 0.1, 25).index).toBe(25);
});
it('rejects short or stationary drawings', () => {
  expect(() =>
    drawnRoute([
      [0, 0],
      [0, 0],
    ]),
  ).toThrow();
  expect(() =>
    drawnRoute([
      [0, 0],
      [1, 0],
      [2, 0],
    ]),
  ).toThrow();
});
it('discards a stale prediction after reset and after a mode change', async () => {
  let finish!: (value: Prediction & { logits: number[] }) => void;
  const policy: Policy = {
    predict: () =>
      new Promise((resolve) => {
        finish = resolve;
      }),
    dispose() {},
  };
  const prediction = {
    steer: 0.5,
    probabilities: Array(15).fill(1 / 15),
    attention: Array(64).fill(1 / 64),
    milliseconds: 5,
    logits: Array(15).fill(0),
  };
  for (const action of ['reset', 'switchMode'] as const) {
    const game = new Game(route, policy);
    game.mode = 'MANUAL';
    game.switchMode();
    const pending = game.advance(0.02);
    game[action]();
    const state = { ...game.state };
    finish(prediction);
    await pending;
    expect(game.targetSteer).toBe(0);
    expect(game.state).toEqual(state);
  }
});
it('uses the same simulated controller cadence at 1× and 16×', async () => {
  const calls: number[] = [];
  const policy: Policy = {
    async predict() {
      calls.push(1);
      return {
        steer: 0,
        probabilities: Array(15).fill(1 / 15),
        attention: Array(64).fill(1 / 64),
        milliseconds: 0,
        logits: [],
      };
    },
    dispose() {},
  };
  for (const speed of [1, 16]) {
    calls.length = 0;
    const game = new Game(route, policy);
    game.mode = 'MANUAL';
    game.switchMode();
    game.speedMultiplier = speed;
    await game.advance(0.08 / speed);
    expect(calls.length).toBe(1);
    expect(game.simulatedSeconds).toBeCloseTo(0.08, 10);
  }
});
