import { expect, test } from '@playwright/test';
// The test harness is exposed only with ?test, and owns the real policy and simulator.
async function harness(page: import('@playwright/test').Page, expression: string) {
  return page.evaluate(expression);
}
test('policy parity, telemetry, drawing, manual jackknife and autonomous completion', async ({
  page,
}) => {
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(e.message));
  page.on('console', (m) => {
    if (m.type() === 'error' && m.text().includes('WebGPU:')) errors.push(m.text());
  });
  await page.goto('/?test');
  await page.waitForFunction('Boolean(window.jackknifeTest)', undefined, { timeout: 60_000 });
  const parity = await harness(page, 'window.jackknifeTest.parity()');
  if (!Array.isArray(parity)) throw new Error('Parity harness returned no results.');
  for (const result of parity) {
    expect(result.logitError).toBeLessThan(0.001);
    expect(result.attentionError).toBeLessThan(0.001);
  }
  await harness(
    page,
    `Object.assign(window.jackknifeTest.game, {mode:'MANUAL', diagnostics:true})`,
  );
  await page.locator('#pause').click();
  await page.locator('#speed').click();
  await page.locator('#speed').click();
  await page.locator('#world').focus();
  await page.keyboard.down('s');
  await page.keyboard.down('a');
  await expect(page.locator('#alert strong')).toHaveText('JACKKNIFED', { timeout: 15000 });
  await page.keyboard.up('s');
  await page.keyboard.up('a');
  await page.getByRole('button', { name: 'RETRY', exact: true }).click();
  await expect(page.locator('#alert')).toBeHidden();
  await page.getByRole('button', { name: 'DRAW', exact: true }).click();
  await expect(page.locator('#drawer')).not.toHaveAttribute('hidden');
  const box = (await page.locator('#world').boundingBox())!;
  await page.mouse.move(box.x + box.width * 0.25, box.y + box.height * 0.6);
  await page.mouse.down();
  for (let i = 0; i <= 60; i++)
    await page.mouse.move(
      box.x + box.width * (0.25 + i / 120),
      box.y + box.height * (0.6 - 0.2 * Math.sin((i / 60) * Math.PI)),
      { steps: 1 },
    );
  await page.mouse.up();
  await expect(page.locator('#drawing-hint')).toBeHidden();
  await expect(page.locator('#drawer')).not.toHaveAttribute('hidden');
  // Restore the exact desktop default route for a complete closed-loop run.
  await harness(
    page,
    `fetch('/models/track.json').then(r=>r.json()).then(r=>window.jackknifeTest.game.reset(r))`,
  );
  await page.getByRole('button', { name: 'MANUAL', exact: true }).click();
  await expect(page.locator('#speed')).toHaveText('16X');
  await page.getByRole('button', { name: 'OFF', exact: true }).click();
  await expect(page.locator('#policy-panel')).not.toHaveAttribute('hidden');
  await expect(page.locator('#alert strong')).toHaveText('DOCK SECURED', { timeout: 90000 });
  await page.getByRole('button', { name: 'NEW ROUTE', exact: true }).click();
  await expect(page.locator('#alert')).toBeHidden();
  const family = await harness(page, 'window.jackknifeTest.game.route.family');
  expect(typeof family).toBe('string');
  expect(errors).toEqual([]);
});
