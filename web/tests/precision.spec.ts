import { expect, test } from '@playwright/test';

test('FP16 preserves steering accuracy and completes the route with attention rays', async ({
  page,
}) => {
  await page.goto('/?test&precision=fp16');
  await page.waitForFunction('Boolean(window.jackknifeTest)', undefined, { timeout: 60_000 });
  const results = (await page.evaluate('window.jackknifeTest.parity()')) as {
    logitError: number;
    attentionError: number;
    probabilityError: number;
    steeringErrorDegrees: number;
  }[];
  for (const result of results) {
    expect(result.logitError).toBeLessThan(0.015);
    expect(result.attentionError).toBeLessThan(0.005);
    expect(result.probabilityError).toBeLessThan(0.002);
    expect(result.steeringErrorDegrees).toBeLessThan(0.05);
  }
  await page.evaluate(
    'window.jackknifeTest.game.rays = true; window.jackknifeTest.game.setPaused(false)',
  );
  await expect(page.locator('#alert strong')).toHaveText('DOCK SECURED', { timeout: 90_000 });
});
