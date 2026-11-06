import { readFileSync } from 'node:fs';
import { expect, test } from '@playwright/test';
import { PNG } from 'pngjs';
const scenes = JSON.parse(
  readFileSync(new URL('../public/assets/visual-scenes.json', import.meta.url), 'utf8'),
);
for (const scene of scenes) {
  test(`matches the Python renderer: ${scene.name}`, async ({ page }) => {
    await page.setViewportSize({ width: scene.width, height: scene.height });
    await page.goto('/?test&freeze');
    await page.waitForFunction('Boolean(window.jackknifeTest)', undefined, { timeout: 60_000 });
    await page.evaluate((scene) => {
      const target = (window as unknown as { jackknifeTest: { game: Record<string, unknown> } })
        .jackknifeTest.game;
      Object.assign(target, scene, {
        paused: false,
        speed: 0,
        targetSteer: 0,
        status: '',
        inferenceMs: 0,
      });
    }, scene);
    await page.mouse.move(scene.width - 1, scene.height - 1);
    await page.waitForTimeout(150);
    const actual = PNG.sync.read(await page.screenshot());
    const expected = PNG.sync.read(
      readFileSync(new URL(`../public/assets/reference-${scene.name}.png`, import.meta.url)),
    );
    let total = 0,
      changed = 0;
    for (let i = 0; i < actual.data.length; i += 4) {
      let max = 0;
      for (let c = 0; c < 3; c++) {
        const difference = Math.abs(actual.data[i + c] - expected.data[i + c]);
        total += difference;
        max = Math.max(max, difference);
      }
      if (max > 25) changed++;
    }
    const meanError = total / (actual.width * actual.height * 3),
      changedFraction = changed / (actual.width * actual.height);
    console.log(
      `${scene.name}: mean RGB error ${meanError.toFixed(2)}, ${(changedFraction * 100).toFixed(2)}% of pixels differ by >25`,
    );
    // Fonts are shared bitmaps. Geometry edges can differ by a pixel across Canvas/WebGPU/Pygame.
    expect(meanError).toBeLessThan(5);
    expect(changedFraction).toBeLessThan(0.055);
  });
}
