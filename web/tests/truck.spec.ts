import { readFileSync } from 'node:fs';
import { test, expect } from '@playwright/test';
import { PNG } from 'pngjs';
interface Scene {
  name: string;
  width: number;
  height: number;
  scale: number;
  state: Record<string, number>;
  commands: unknown[];
}
const scenes: Scene[] = JSON.parse(
  readFileSync(new URL('../public/assets/truck-scenes.json', import.meta.url), 'utf8'),
);
test('truck-only geometry and pixels match Pygame at every reference articulation', async ({
  page,
}) => {
  await page.goto('/?test&freeze');
  await page.waitForFunction('Boolean(window.jackknifeTest)', undefined, { timeout: 60_000 });
  for (const scene of scenes) {
    const actual = await page.evaluate(async (scene) => {
      const harness = (
        window as unknown as {
          jackknifeTest: {
            renderTruck: (scene: Scene) => Promise<{ png: string; commands: unknown[] }>;
          };
        }
      ).jackknifeTest;
      return harness.renderTruck(scene);
    }, scene);
    expect(actual.commands.length, `${scene.name} draw command count`).toBe(scene.commands.length);
    for (let i = 0; i < scene.commands.length; i++)
      expect(actual.commands[i], `${scene.name} draw ${i}`).toEqual(scene.commands[i]);
    const pixels = PNG.sync.read(Buffer.from(actual.png.split(',')[1], 'base64'));
    const reference = PNG.sync.read(
      readFileSync(new URL(`../public/assets/${scene.name}.png`, import.meta.url)),
    );
    let changed = 0,
      foreground = 0;
    for (let i = 0; i < pixels.data.length; i += 4) {
      if (reference.data[i + 3]) foreground++;
      if ([0, 1, 2, 3].some((c) => pixels.data[i + c] !== reference.data[i + c])) changed++;
    }
    console.log(`${scene.name}: ${changed} differing pixels / ${foreground} truck pixels`);
    expect(changed, `${scene.name} isolated truck pixel mismatch`).toBe(0);
  }
});
