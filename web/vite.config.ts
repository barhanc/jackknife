import { defineConfig } from 'vite';
import typegpu from 'unplugin-typegpu/vite';
export default defineConfig({ base: './', plugins: [typegpu()], worker: { format: 'es' } });
