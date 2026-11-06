import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { wrap } from '../src/simulation/physics';
import {
  generateFamily,
  sampleRoute,
  randomSource,
  type Family,
  type RandomSource,
} from '../src/simulation/generator';
interface Event {
  kind: string;
  value: number | number[];
  low?: number;
  high?: number;
}
interface Reference {
  family: Family;
  length: number;
  tape: Event[];
  points: number[][];
  headings: number[];
  curvatures: number[];
}
const references: Reference[] = JSON.parse(
  readFileSync(new URL('../public/models/path-reference.json', import.meta.url), 'utf8'),
);
describe('the same procedural families as Python', () => {
  for (const reference of references)
    it(reference.family, () => {
      let cursor = 0;
      const take = (kind: string, a?: number, b?: number): number | number[] => {
        const event = reference.tape[cursor++];
        expect(event.kind).toBe(kind);
        if (a !== undefined) {
          expect(event.low).toBeCloseTo(a, 12);
          expect(event.high).toBeCloseTo(b!, 12);
        }
        return event.value;
      };
      const scalar = (kind: string, a?: number, b?: number) => {
        const value = take(kind, a, b);
        if (typeof value !== 'number') throw new Error('Expected scalar random draw');
        return value;
      };
      const rng: RandomSource = {
        uniform: (a, b) => scalar('uniform', a, b),
        integer: (a, b) => scalar('integer', a, b),
        random: () => scalar('random'),
        sign: () => scalar('sign'),
        sample: () => {
          const value = take('sample');
          if (!Array.isArray(value)) throw new Error('Expected sampled indices');
          return value;
        },
      };
      const route = generateFamily(reference.family, reference.length, rng);
      expect(cursor).toBe(reference.tape.length);
      const countDelta = Math.abs(route.points.length - reference.points.length);
      expect(countDelta).toBeLessThanOrEqual(1);
      // ceil(length / .8) can straddle an integer by one floating-point ulp.
      // In that case compare the same arc-length fractions rather than array indices.
      const interpolate = (values: number[], fraction: number) => {
        const position = fraction * (values.length - 1),
          left = Math.floor(position),
          right = Math.min(values.length - 1, left + 1);
        return values[left] + (values[right] - values[left]) * (position - left);
      };
      let maximum = 0;
      for (let i = 0; i < route.points.length; i++) {
        const fraction = i / (route.points.length - 1);
        for (let j = 0; j < 2; j++)
          maximum = Math.max(
            maximum,
            Math.abs(
              route.points[i][j] -
                interpolate(
                  reference.points.map((p) => p[j]),
                  fraction,
                ),
            ),
          );
        const position = fraction * (reference.headings.length - 1),
          left = Math.floor(position),
          right = Math.min(reference.headings.length - 1, left + 1);
        const expectedHeading =
          reference.headings[left] +
          wrap(reference.headings[right] - reference.headings[left]) * (position - left);
        expect(Math.abs(wrap(route.headings[i] - expectedHeading))).toBeLessThan(
          countDelta ? 0.005 : 1e-5,
        );
        expect(
          Math.abs(route.curvatures[i] - interpolate(reference.curvatures, fraction)),
        ).toBeLessThan(countDelta ? 0.005 : 1e-5);
      }
      expect(maximum).toBeLessThan(countDelta ? 0.015 : 1e-5);
    });
});
it('generates varied lot-fitting routes rather than reflected copies', () => {
  let seed = 9124;
  const rng = randomSource(() => {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    return seed / 2 ** 32;
  });
  const routes = Array.from({ length: 40 }, () => sampleRoute(rng));
  expect(new Set(routes.map((r) => r.family)).size).toBeGreaterThan(10);
  for (const route of routes) {
    expect(route.points.length).toBeGreaterThan(20);
    expect(route.points.flat().every(Number.isFinite)).toBe(true);
    expect(Math.max(...route.points.map((p) => Math.abs(p[0])))).toBeLessThanOrEqual(70);
    expect(Math.max(...route.points.map((p) => Math.abs(p[1])))).toBeLessThanOrEqual(40);
  }
});
