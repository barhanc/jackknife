/** Paint all attention rays with integer pixels and one Canvas image submission. */
import type { Point } from '../simulation/physics';
import { strokeLine } from './pixelRaster';
export function createAttentionLayer() {
  const canvas = document.createElement('canvas');
  const context = canvas.getContext('2d', { willReadFrequently: true })!;
  let image: ImageData | undefined;
  let pixels = new Uint32Array(0);
  return {
    draw(target: CanvasRenderingContext2D, from: Point, rays: { point: Point; weight: number }[]) {
      if (!rays.length) return;
      const points = [from, ...rays.map((ray) => ray.point)];
      const left = Math.max(0, Math.min(...points.map((p) => p[0])) - 2);
      const top = Math.max(0, Math.min(...points.map((p) => p[1])) - 2);
      const right = Math.min(target.canvas.width, Math.max(...points.map((p) => p[0])) + 3);
      const bottom = Math.min(target.canvas.height, Math.max(...points.map((p) => p[1])) + 3);
      if (right <= left || bottom <= top) return;
      const width = Math.ceil((right - left) / 32) * 32,
        height = Math.ceil((bottom - top) / 32) * 32;
      if (!image || width > image.width || height > image.height) {
        canvas.width = Math.max(width, image?.width ?? 0);
        canvas.height = Math.max(height, image?.height ?? 0);
        image = new ImageData(canvas.width, canvas.height);
        pixels = new Uint32Array(image.data.buffer);
      }
      pixels.fill(0);
      for (const ray of rays) {
        for (const [width, r, g, b, strength] of [
          [2, 45, 180, 225, 55],
          [1, 185, 235, 255, 135],
        ]) {
          const rgba = (Math.trunc(ray.weight * strength) << 24) | (b << 16) | (g << 8) | r;
          strokeLine(from, ray.point, width, (y, a, b) => {
            const row = y - top,
              x0 = Math.max(a - left, 0),
              x1 = Math.min(b - left, image!.width - 1);
            if (row < 0 || row >= image!.height || x1 < x0) return;
            pixels.fill(rgba, row * image!.width + x0, row * image!.width + x1 + 1);
          });
        }
      }
      context.putImageData(image, 0, 0);
      target.drawImage(canvas, left, top);
    },
  };
}
