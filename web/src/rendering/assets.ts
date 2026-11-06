import { fillCircle, fillPolygon, strokeLine } from './pixelRaster';
import type { Point } from '../simulation/physics';
/** Desktop-generated bitmaps preserve Pygame's fonts, snow seed and light fields. */
export type Rect = { x: number; y: number; w: number; h: number };
export const rect = (x = 0, y = 0, w = 0, h = 0): Rect => ({
  x: Math.trunc(x),
  y: Math.trunc(y),
  w: Math.trunc(w),
  h: Math.trunc(h),
});
export const move = (r: Rect, dx: number, dy: number) => rect(r.x + dx, r.y + dy, r.w, r.h);
export const inset = (r: Rect, x: number, y = x) =>
  rect(r.x + x, r.y + y, r.w - 2 * x, r.h - 2 * y);
export const contains = (r: Rect, x: number, y: number) =>
  x >= r.x && y >= r.y && x < r.x + r.w && y < r.y + r.h;
export type RGB = readonly [number, number, number];
export const css = (color: RGB, alpha = 1) => `rgba(${color.join(',')},${alpha})`;
interface Glyph {
  rect: [number, number, number, number];
  ink: [number, number, number, number];
  advance: number;
}
interface Font {
  height: number;
  glyphs: Record<string, Glyph>;
}
export interface Scenery {
  slabs: [number, number, RGB][];
  patches: [number, number, number, number, number][];
  grain: [number, number, number][];
  stains: [number, number, number][];
  flakes: [number, number, number, number][];
  lights: Record<string, { origin: [number, number]; size: [number, number] }>;
}
const image = (url: string): Promise<HTMLImageElement> =>
  new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () =>
      reject(new Error(`Missing desktop asset: ${url}. Run npm run export-model.`));
    img.src = url;
  });
export async function loadAssets() {
  const base = `${import.meta.env.BASE_URL}assets/`;
  const [atlas, fonts, scenery, snow, snowShadow, ...lights] = await Promise.all([
    image(base + 'fonts.png'),
    fetch(base + 'fonts.json').then((r) => r.json()) as Promise<Record<string, Font>>,
    fetch(base + 'scenery.json').then((r) => r.json()) as Promise<Scenery>,
    image(base + 'snow.png'),
    image(base + 'snow-shadow.png'),
    ...['head', 'reverse', 'white', 'red'].map((name) => image(`${base}light-${name}.png`)),
  ]);
  const cache = new Map<string, HTMLCanvasElement>();
  const font = (key: string) => {
    const value = fonts[key];
    if (!value) throw new Error(`Missing desktop font ${key}`);
    return value;
  };
  const measure = (text: string, key: string) =>
    [...text].reduce((sum, c) => sum + (font(key).glyphs[c] ?? font(key).glyphs['?']).advance, 0);
  const glyph = (text: string, key: string, color: RGB) => {
    const id = `${key}/${color}/${text}`;
    const cached = cache.get(id);
    if (cached) return cached;
    const f = font(key);
    let pen = 0,
      left = Infinity,
      top = Infinity,
      right = 0,
      bottom = 0;
    for (const char of text) {
      const g = f.glyphs[char] ?? f.glyphs['?'];
      const [x, y, w, h] = g.ink;
      if (w && h) {
        left = Math.min(left, pen + x);
        top = Math.min(top, y);
        right = Math.max(right, pen + x + w);
        bottom = Math.max(bottom, y + h);
      }
      pen += g.advance;
    }
    if (!Number.isFinite(left)) {
      left = top = 0;
      right = measure(text, key);
      bottom = f.height;
    }
    const canvas = document.createElement('canvas');
    canvas.width = Math.max(1, right - left);
    canvas.height = Math.max(1, bottom - top);
    const ctx = canvas.getContext('2d')!;
    ctx.imageSmoothingEnabled = false;
    pen = 0;
    for (const char of text) {
      const g = f.glyphs[char] ?? f.glyphs['?'];
      ctx.drawImage(atlas, ...g.rect, pen - left, -top, g.rect[2], g.rect[3]);
      pen += g.advance;
    }
    ctx.globalCompositeOperation = 'source-in';
    ctx.fillStyle = css(color);
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    if (cache.size > 1024) cache.clear();
    cache.set(id, canvas);
    return canvas;
  };
  return {
    scenery,
    snow,
    snowShadow,
    lightImages: Object.fromEntries(
      ['head', 'reverse', 'white', 'red'].map((name, i) => [name, lights[i]]),
    ),
    measure,
    glyph,
    text(
      ctx: CanvasRenderingContext2D,
      text: string,
      key: string,
      r: Rect,
      color: RGB,
      center = false,
    ) {
      let surface = glyph(text, key, color);
      while (text && surface.width > r.w) {
        text = text.slice(0, -1);
        surface = glyph(text, key, color);
      }
      ctx.drawImage(
        surface,
        Math.trunc(center ? r.x + Math.trunc(r.w / 2) - Math.trunc(surface.width / 2) : r.x),
        Math.trunc(r.y + Math.trunc(r.h / 2) - Math.trunc(surface.height / 2)),
      );
    },
  };
}
export type Assets = Awaited<ReturnType<typeof loadAssets>>;
export function polygon(
  ctx: CanvasRenderingContext2D,
  points: readonly (readonly number[])[],
  color: RGB,
  width = 0,
  alpha = 1,
) {
  const pixels: Point[] = points.map((p) => [Math.trunc(p[0]), Math.trunc(p[1])]);
  ctx.fillStyle = css(color, alpha);
  const span = (y: number, left: number, right: number) =>
    ctx.fillRect(left, y, right - left + 1, 1);
  if (width)
    for (let i = 0; i < pixels.length; i++)
      strokeLine(pixels[i], pixels[(i + 1) % pixels.length], Math.max(1, Math.trunc(width)), span);
  else fillPolygon(pixels, span);
}
export function line(
  ctx: CanvasRenderingContext2D,
  a: readonly number[],
  b: readonly number[],
  color: RGB,
  width = 1,
  alpha = 1,
) {
  ctx.fillStyle = css(color, alpha);
  strokeLine(
    [Math.trunc(a[0]), Math.trunc(a[1])],
    [Math.trunc(b[0]), Math.trunc(b[1])],
    Math.max(1, Math.trunc(width)),
    (y, left, right) => ctx.fillRect(left, y, right - left + 1, 1),
  );
}

export function fill(ctx: CanvasRenderingContext2D, r: Rect, color: RGB) {
  ctx.fillStyle = css(color);
  ctx.fillRect(r.x, r.y, r.w, r.h);
}

export function circle(
  ctx: CanvasRenderingContext2D,
  point: readonly number[],
  radius: number,
  color: RGB,
  alpha = 1,
) {
  ctx.fillStyle = css(color, alpha);
  fillCircle(Math.trunc(point[0]), Math.trunc(point[1]), Math.trunc(radius), (y, left, right) =>
    ctx.fillRect(left, y, right - left + 1, 1),
  );
}
