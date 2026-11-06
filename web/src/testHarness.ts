/** Explicit ?test mode for numerical parity and browser integration tests. */
import type { Policy } from './inference/policy';
import type { Game } from './simulation/game';
import { drawnRoute } from './simulation/path';
export function attachHarness(game: Game, policy: Policy) {
  Object.assign(window, {
    jackknifeTest: {
      game,
      drawnRoute,
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
          results.push({
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
