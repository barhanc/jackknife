/** SPDX-License-Identifier: LGPL-2.1-or-later
 * Pixel raster routines adapted from Pygame 2.6.1 draw.c.
 * Copyright (C) 2000-2001 Pete Shinners and Pygame contributors.
 * See public/licenses/pygame-LGPL.txt for the license.
 */
/** Integer scanline fills and pixel-width lines compatible with Pygame's draw conventions.
 * Algorithm reference: https://github.com/pygame/pygame/blob/2.6.1/src_c/draw.c
 */
import type { Point } from '../simulation/physics';
export type Span = (y: number, left: number, right: number) => void;
export function fillPolygon(points: Point[], span: Span) {
  const top = Math.min(...points.map((p) => p[1])),
    bottom = Math.max(...points.map((p) => p[1]));
  if (top === bottom) {
    span(top, Math.min(...points.map((p) => p[0])), Math.max(...points.map((p) => p[0])));
    return;
  }
  for (let y = top; y <= bottom; y++) {
    const crossings: number[] = [];
    for (let i = 0; i < points.length; i++) {
      let a = points[(i + points.length - 1) % points.length],
        b = points[i];
      if (a[1] === b[1]) continue;
      if (a[1] > b[1]) [a, b] = [b, a];
      if (y >= a[1] && (y < b[1] || (y === bottom && b[1] === bottom)))
        crossings.push(Math.trunc(a[0] + ((y - a[1]) * (b[0] - a[0])) / (b[1] - a[1])));
    }
    crossings.sort((a, b) => a - b);
    for (let i = 0; i + 1 < crossings.length; i += 2) span(y, crossings[i], crossings[i + 1]);
  }
  for (let i = 0; i < points.length; i++) {
    const a = points[i],
      b = points[(i + points.length - 1) % points.length];
    if (a[1] === b[1] && a[1] > top && a[1] < bottom)
      span(a[1], Math.min(a[0], b[0]), Math.max(a[0], b[0]));
  }
}
export function strokeLine(start: Point, end: Point, width: number, span: Span) {
  let [x, y] = start;
  const [targetX, targetY] = end,
    dx = Math.abs(targetX - x),
    dy = Math.abs(targetY - y),
    sx = x < targetX ? 1 : -1,
    sy = y < targetY ? 1 : -1;
  const horizontalThickness = dx <= dy,
    low = Math.floor((width - 1) / 2),
    high = Math.floor(width / 2);
  if (!dx && !dy) {
    span(y, x - low, x + high);
    return;
  }
  if (!dy) {
    for (let row = y - low; row <= y + high; row++)
      span(row, Math.min(x, targetX), Math.max(x, targetX));
    return;
  }
  if (!dx) {
    for (let row = Math.min(y, targetY); row <= Math.max(y, targetY); row++)
      span(row, x - low, x + high);
    return;
  }
  let error = Math.trunc((dx > dy ? dx : -dy) / 2);
  const emit = () => {
    if (horizontalThickness) span(y, x - low, x + high);
    else for (let row = y - low; row <= y + high; row++) span(row, x, x);
  };
  // Thick Pygame lines stop on the major axis; flat ends keep their full width.
  while (
    width === 1
      ? x !== targetX || y !== targetY
      : horizontalThickness
        ? y !== targetY
        : x !== targetX
  ) {
    emit();
    const previous = error;
    if (previous > -dx) {
      error -= dy;
      x += sx;
    }
    if (previous < dy) {
      error += dx;
      y += sy;
    }
  }
  emit();
}

/** Filled midpoint circle with Pygame's even-diameter pixel convention. */
export function fillCircle(cx: number, cy: number, radius: number, span: Span) {
  let decision = 1 - radius,
    deltaX = 0,
    deltaY = -2 * radius,
    x = 0,
    y = radius;
  while (x < y) {
    if (decision >= 0) {
      y--;
      deltaY += 2;
      decision += deltaY;
    }
    x++;
    deltaX += 2;
    decision += deltaX + 1;
    if (decision >= 0) {
      span(cy + y - 1, cx - x, cx + x - 1);
      span(cy - y, cx - x, cx + x - 1);
    }
    span(cy + x - 1, cx - y, cx + y - 1);
    span(cy - x, cx - y, cx + y - 1);
  }
}
