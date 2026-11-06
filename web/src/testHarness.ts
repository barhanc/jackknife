/** Explicit ?test mode for numerical parity and browser integration tests. */
import type { TgpuRoot } from 'typegpu';
import { Mesh, createPainter, type Color } from './rendering/primitives';
import { vehicle } from './rendering/vehicle';
import type { Point, State } from './simulation/physics';
import type { Policy } from './inference/policy';
import type { Game } from './simulation/game';
import { drawnRoute } from './simulation/path';
export function attachHarness(game: Game, policy: Policy, root: TgpuRoot) {
  Object.assign(window, {
    jackknifeTest: {
      game,
      drawnRoute,
      async renderTruck(sample: { width: number; height: number; scale: number; state: State }) {
        const canvas = document.createElement('canvas');
        canvas.width = sample.width;
        canvas.height = sample.height;
        const commands: { kind: string; points: Point[]; color: Color; width?: number }[] = [];
        class Capture extends Mesh {
          override polygon(points: Point[], color: Color) {
            commands.push({ kind: 'polygon', points: points.map((p) => this.project(p)), color });
            super.polygon(points, color);
          }
          override line(a: Point, b: Point, width: number, color: Color) {
            commands.push({
              kind: 'line',
              points: [this.project(a), this.project(b)],
              color,
              width,
            });
            super.line(a, b, width, color);
          }
        }
        const mesh = new Capture(sample.scale, sample.width, sample.height);
        const painter = createPainter(root, canvas);
        vehicle(mesh, sample.state);
        painter.draw(mesh);
        await root.device.queue.onSubmittedWorkDone();
        const png = canvas.toDataURL('image/png');
        painter.dispose();
        return { png, commands };
      },
      async inferenceDuringRenderBlock() {
        game.setPaused(true);
        const sample = (await (await fetch(`${import.meta.env.BASE_URL}models/parity.json`)).json())
          .cases[0];
        const tokens = new Float32Array(sample.tokens.flat()),
          hitch = new Float32Array(sample.hitch);
        const baseline = await policy.predict(tokens, hitch);
        const blockedFor = Math.max(150, Math.min(1000, baseline.milliseconds * 5));
        const start = performance.now();
        const request = policy.predict(
          new Float32Array(sample.tokens.flat()),
          new Float32Array(sample.hitch),
        );
        while (performance.now() - start < blockedFor) {
          /* Deliberately block only the rendering thread. */
        }
        const result = await request;
        return {
          milliseconds: result.milliseconds,
          mainThreadLatency: performance.now() - start,
          baseline: baseline.milliseconds,
          blockedFor,
        };
      },
      async benchmark(count = 200) {
        game.setPaused(true);
        const fixtures = await (
          await fetch(`${import.meta.env.BASE_URL}models/parity.json`)
        ).json();
        const samples = fixtures.cases.map((sample: { tokens: number[][]; hitch: number[] }) => ({
          tokens: new Float32Array(sample.tokens.flat()),
          hitch: new Float32Array(sample.hitch),
        }));
        const times = [];
        for (let i = 0; i < count + 20; i++) {
          const sample = samples[i % samples.length];
          const result = await policy.predict(sample.tokens, sample.hitch);
          if (i >= 20) times.push(result.milliseconds);
        }
        times.sort((a, b) => a - b);
        return {
          count,
          median: times[Math.floor(count / 2)],
          p95: times[Math.ceil(count * 0.95) - 1],
          mean: times.reduce((a, b) => a + b, 0) / count,
        };
      },
      async parity() {
        const response = await fetch(`${import.meta.env.BASE_URL}models/parity.json`);
        const fixtures = await response.json();
        game.setPaused(true);
        const results = [];
        for (const sample of fixtures.cases) {
          const prediction = await policy.predict(
            new Float32Array(sample.tokens.flat()),
            new Float32Array(sample.hitch),
          );
          const maximum = Math.max(...sample.logits);
          const exponentials = sample.logits.map((x: number) => Math.exp(x - maximum));
          const sum = exponentials.reduce((a: number, b: number) => a + b, 0);
          const probabilities = exponentials.map((x: number) => x / sum);
          const steering = probabilities.reduce(
            (total: number, p: number, i: number) => total + p * (-35 + 5 * i),
            0,
          );
          results.push({
            probabilityError: Math.max(
              ...prediction.probabilities.map((x, i) => Math.abs(x - probabilities[i])),
            ),
            steeringErrorDegrees: Math.abs((prediction.steer * 180) / Math.PI - steering),
            logitError: Math.max(
              ...prediction.logits.map((x, i) => Math.abs(x - sample.logits[i])),
            ),
            attentionError: Math.max(
              ...prediction.attention.map((x, i) => Math.abs(x - sample.attention[i])),
            ),
            ms: prediction.milliseconds,
          });
        }
        return results;
      },
    },
  });
}
