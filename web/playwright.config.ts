import { defineConfig } from '@playwright/test';
export default defineConfig({
  testDir: './tests',
  testMatch: '**/*.spec.ts',
  timeout: 120_000,
  workers: 1,
  use: {
    channel: 'chromium',
    baseURL: 'http://127.0.0.1:5173',
    viewport: { width: 1440, height: 950 },
    launchOptions: { args: ['--enable-unsafe-webgpu', '--ignore-gpu-blocklist'] },
  },
  webServer: {
    command: 'npm run dev',
    url: 'http://127.0.0.1:5173',
    reuseExistingServer: !process.env.CI,
  },
});
