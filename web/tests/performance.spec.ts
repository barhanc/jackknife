import { expect, test } from '@playwright/test';

test('inference timing excludes a blocked rendering thread', async ({ page }) => {
  await page.goto('/?test');
  await page.waitForFunction('Boolean(window.jackknifeTest)', undefined, { timeout: 60_000 });
  const result = await page.evaluate(async () => {
    const harness = (
      window as unknown as {
        jackknifeTest: {
          inferenceDuringRenderBlock: () => Promise<{
            milliseconds: number;
            mainThreadLatency: number;
            baseline: number;
            blockedFor: number;
          }>;
        };
      }
    ).jackknifeTest;
    return harness.inferenceDuringRenderBlock();
  });
  expect(result.mainThreadLatency).toBeGreaterThanOrEqual(result.blockedFor);
  expect(result.milliseconds).toBeLessThan(result.mainThreadLatency * 0.75);
});

test('telemetry and rays keep Canvas reads on CPU masks', async ({ page }) => {
  const warnings: string[] = [];
  page.on('console', (message) => {
    if (message.text().includes('willReadFrequently')) warnings.push(message.text());
  });
  await page.addInitScript(() => {
    const counters = { total: 0, unhinted: 0 };
    Object.assign(window, { maskReadbacks: counters });
    const original = CanvasRenderingContext2D.prototype.getImageData;
    CanvasRenderingContext2D.prototype.getImageData = function (x, y, w, h, settings) {
      counters.total++;
      if (!this.getContextAttributes().willReadFrequently) counters.unhinted++;
      return original.call(this, x, y, w, h, settings);
    };
  });
  await page.goto('/?test');
  await page.waitForFunction('Boolean(window.jackknifeTest)', undefined, { timeout: 60_000 });
  await page.evaluate(() => {
    const game = (
      window as unknown as {
        jackknifeTest: {
          game: { mode: string; speedMultiplier: number; setPaused: (paused: boolean) => void };
        };
      }
    ).jackknifeTest.game;
    if (game.mode !== 'AUTONOMOUS' || game.speedMultiplier !== 16)
      throw new Error('Startup must be AUTO at 16x.');
    game.setPaused(false);
  });
  for (let i = 0; i < 4; i++) {
    await page.locator('#telemetry').click();
    await page.waitForTimeout(100);
  }
  await page.locator('#rays').click();
  await page.waitForTimeout(250);
  const counters = await page.evaluate(
    () =>
      (window as unknown as { maskReadbacks: { total: number; unhinted: number } }).maskReadbacks,
  );
  expect(counters.total).toBeGreaterThan(0);
  expect(counters.unhinted).toBe(0);
  expect(warnings).toEqual([]);
  const favicon = await page.locator('link[rel="icon"]').getAttribute('href');
  const response = await page.request.get(favicon!);
  expect(response.ok()).toBe(true);
  expect(await response.text()).toContain('<title>Pixel truck</title>');
});
