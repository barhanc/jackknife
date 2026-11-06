/** Direct port of src/game/hud.py; bitmap text and pixel layout follow the desktop HUD. */
import type { Game } from '../simulation/game';
import {
  type Assets,
  type Rect,
  type RGB,
  rect,
  move,
  inset,
  contains,
  fill,
  line,
  polygon,
} from '../rendering/assets';
const PANEL: RGB = [21, 22, 23],
  BORDER: RGB = [72, 73, 72],
  TEXT: RGB = [233, 233, 226],
  MUTED: RGB = [161, 164, 161],
  AMBER: RGB = [255, 190, 73],
  GREEN: RGB = [135, 225, 130],
  RED: RGB = [255, 85, 112];
export interface HudLayout {
  width: number;
  height: number;
  scale: number;
  pad: number;
  gap: number;
  toolbar: Rect;
  yard: Rect;
  drawer: Rect;
  footer: Rect;
  horizontal: boolean;
}
export function createHud(app: HTMLElement) {
  app.innerHTML = `<canvas id="world" tabindex="0" aria-label="Truck simulator. Drive with WASD or draw a route with the mouse."></canvas><canvas id="hud-canvas" aria-hidden="true"></canvas><div id="hit-targets">${['mode', 'draw', 'new', 'speed', 'pause', 'telemetry', 'cancel', 'reset', 'rays', 'alert-reset'].map((id) => `<button id="${id}"></button>`).join('')}</div><div id="alert" class="sr-only" role="status" hidden><strong></strong><span></span></div><div id="drawing-hint" class="sr-only" hidden><span></span></div><div id="drawer" class="sr-only">Diagnostics</div><div id="policy-panel" class="sr-only"></div><div id="loading" role="status"><strong>JACKKNIFE</strong><p>Starting WebGPU…</p></div>`;
  const get = <T extends HTMLElement = HTMLElement>(id: string) => app.querySelector<T>(`#${id}`)!;
  const canvas = get<HTMLCanvasElement>('world'),
    overlay = get<HTMLCanvasElement>('hud-canvas'),
    ctx = overlay.getContext('2d')!;
  let assets: Assets,
    current: HudLayout,
    scroll = 0,
    limit = 0,
    pointer: [number, number] = [-1, -1];
  let specs: {
    id: string;
    r: Rect;
    label: string;
    key: string;
    active?: boolean;
    primary?: boolean;
  }[] = [];
  app.addEventListener('pointermove', (e) => {
    pointer = [e.clientX, e.clientY];
  });
  app.addEventListener(
    'wheel',
    (e) => {
      if (current && contains(current.drawer, e.clientX, e.clientY)) {
        e.preventDefault();
        scroll = Math.max(0, Math.min(limit, scroll + Math.sign(e.deltaY) * 36 * current.scale));
      }
    },
    { passive: false },
  );
  const flow = (
    widths: number[],
    left: number,
    top: number,
    right: number,
    height: number,
    gap: number,
  ) => {
    let x = left,
      y = top;
    const cells = widths.map((w) => {
      if (x > left && x + w > right) {
        x = left;
        y += height + gap;
      }
      const r = rect(x, y, Math.min(w, Math.max(1, right - left)), height);
      x += w + gap;
      return r;
    });
    return { cells, bottom: y + height };
  };
  const font = (family: string, size: number) =>
    `${family}:${Math.max(8, Math.min(48, Math.trunc(size)))}`;
  const arcade = (s: number) => font('arcade', 10 * s),
    small = (s: number) => font('arcade', 8 * s),
    sm = (s: number) => font('mono', 11 * s);
  const panel = (r: Rect) => {
    fill(ctx, r, PANEL);
    for (let y = r.y + 2; y < r.y + r.h; y += Math.max(4, Math.trunc(4 * current.scale)))
      line(ctx, [r.x, y], [r.x + r.w - 1, y], [25, 26, 27]);
  };
  const corners = (r: Rect, cut = 4) => [
    [r.x + cut, r.y],
    [r.x + r.w - cut - 1, r.y],
    [r.x + r.w - 1, r.y + cut],
    [r.x + r.w - 1, r.y + r.h - cut - 1],
    [r.x + r.w - cut - 1, r.y + r.h - 1],
    [r.x + cut, r.y + r.h - 1],
    [r.x, r.y + r.h - cut - 1],
    [r.x, r.y + cut],
  ];
  const text = (
    value: string,
    r: Rect,
    key = font('mono', 13 * current.scale),
    color: RGB = TEXT,
    center = false,
  ) => assets.text(ctx, value, key, r, color, center);
  const button = (
    id: string,
    r: Rect,
    label: string,
    key = '',
    active = false,
    primary = false,
  ) => {
    const s = current.scale,
      gap = current.gap,
      hover = contains(r, ...pointer);
    polygon(
      ctx,
      corners(r),
      primary ? [49, 43, 31] : hover || active ? [47, 49, 49] : [29, 30, 31],
    );
    polygon(ctx, corners(r), primary || hover || active ? AMBER : BORDER, 1);
    line(
      ctx,
      [r.x + 5, r.y + r.h - 3],
      [r.x + r.w - 6, r.y + r.h - 3],
      primary ? [104, 83, 45] : [58, 60, 60],
    );
    let target = inset(r, gap, 0);
    if (key) {
      const kw = assets.measure(key, sm(s)) + gap,
        lw = assets.glyph(label, arcade(s), TEXT).width,
        group = lw + gap + kw;
      const left = r.x + Math.trunc(r.w / 2) - Math.trunc(group / 2),
        kr = rect(left + lw + gap, r.y + Math.trunc(r.h / 2) - Math.trunc(10 * s), kw, 20 * s);
      fill(ctx, kr, [12, 13, 14]);
      text(key, kr, sm(s), MUTED, true);
      target = rect(left, r.y, lw, r.h);
    }
    text(label, target, arcade(s), primary ? AMBER : TEXT, true);
    specs.push({ id, r, label, key, active, primary });
  };
  const border = (r: Rect, color: RGB, width = 1) =>
    polygon(
      ctx,
      [
        [r.x, r.y],
        [r.x + r.w - 1, r.y],
        [r.x + r.w - 1, r.y + r.h - 1],
        [r.x, r.y + r.h - 1],
      ],
      color,
      width,
    );
  const layout = (game: Game) => {
    const width = app.clientWidth,
      height = app.clientHeight,
      s = Math.max(1, Math.min(width / 1400, height / 900));
    const pad = Math.max(8, Math.trunc(16 * s)),
      gap = Math.max(6, Math.trunc(8 * s)),
      row = Math.trunc(40 * s);
    const toolbarSpecs = [
      ['mode', 'MANUAL', 'M'],
      ['draw', 'DRAW', 'E'],
      ['new', 'NEW ROUTE', 'P'],
      ['speed', '16X', 'T'],
      ['pause', 'CONTINUE', 'Space'],
      ['telemetry', 'TELEMETRY', 'H'],
    ];
    const widths = [
      assets.measure('JACKKNIFE', font('arcade', 18 * s)) + Math.trunc(24 * s),
      ...toolbarSpecs.map(
        ([, label, key]) => assets.measure(label, arcade(s)) + assets.measure(key, sm(s)) + 4 * gap,
      ),
    ];
    const top = flow(widths, pad, pad, width - pad, row, gap);
    if (top.cells.at(-1)!.y === top.cells[0].y) {
      const extra = width - pad - top.cells.at(-1)!.x - top.cells.at(-1)!.w;
      top.cells = top.cells.map((r, i) => (i ? move(r, extra, 0) : r));
    }
    const sw = [
      assets.measure('AUTONOMOUS', arcade(s)) + Math.trunc(20 * s),
      assets.measure('STABILITY', small(s)) + Math.trunc(128 * s),
      assets.measure('INFERENCE', small(s)) +
        assets.measure('999.9 ms', font('mono', 13 * s)) +
        2 * gap,
      assets.measure('RESET', arcade(s)) + assets.measure('R', sm(s)) + 4 * gap,
    ];
    const bottom = flow(sw, pad, gap, width - pad, Math.trunc(32 * s), gap),
      footer = rect(0, height - bottom.bottom - gap, width, bottom.bottom + gap);
    bottom.cells = bottom.cells.map((r) => move(r, 0, footer.y));
    if (bottom.cells.at(-1)!.y === bottom.cells[0].y) {
      const extra = width - 2 * pad - sw.reduce((a, b) => a + b, 0) - 3 * gap;
      bottom.cells = bottom.cells.map((r, i) => move(r, Math.round((i * extra) / 3), 0));
    }
    const toolbar = rect(0, 0, width, top.bottom + pad),
      yard = rect(0, toolbar.h, width, Math.max(1, footer.y - toolbar.h));
    const horizontal = width < Math.trunc(900 * s),
      drawer = rect();
    if (game.diagnostics) {
      if (horizontal) {
        const h = Math.min(Math.trunc(250 * s), Math.floor(yard.h / 2));
        Object.assign(drawer, rect(0, yard.y + yard.h - h, width, h));
        yard.h -= h;
      } else {
        const w = Math.trunc(300 * s);
        Object.assign(drawer, rect(width - w, yard.y, w, yard.h));
        yard.w -= w;
      }
    }
    current = { width, height, scale: s, pad, gap, toolbar, yard, drawer, footer, horizontal };
    return { top: top.cells, bottom: bottom.cells, toolbarSpecs };
  };
  const diagnostics = (game: Game) => {
    const { drawer: area, pad, gap, scale: s, width, horizontal: drawerHorizontal } = current;
    panel(area);
    border(area, BORDER);
    ctx.save();
    ctx.beginPath();
    ctx.rect(area.x, area.y, area.w, area.h);
    ctx.clip();
    const inner = inset(area, pad),
      heading = rect(inner.x, inner.y, inner.w, 24 * s),
      body = rect(
        inner.x,
        inner.y + Math.trunc(36 * s),
        inner.w,
        Math.max(1, inner.h - Math.trunc(36 * s)),
      );
    const horizontal = drawerHorizontal && width >= 600,
      sectionW = horizontal ? Math.floor(inner.w / 2) - gap : inner.w;
    const description =
      'Lines show which path points the AI is looking at. Brighter lines mean more attention.';
    const lines: string[] = [];
    let currentLine = '';
    for (const word of description.split(' ')) {
      const candidate = `${currentLine} ${word}`.trim();
      if (currentLine && assets.measure(candidate, sm(s)) > sectionW) {
        lines.push(currentLine);
        currentLine = word;
      } else currentLine = candidate;
    }
    if (currentLine) lines.push(currentLine);
    const chartH = Math.trunc(128 * s),
      raysH = Math.trunc(44 * s) + lines.length * Math.trunc(16 * s),
      controlsH = Math.trunc(256 * s),
      leftH = chartH + Math.trunc(20 * s) + raysH;
    const required = Math.max(
      body.h,
      horizontal ? Math.max(leftH, controlsH) : leftH + Math.trunc(24 * s) + controlsH,
    );
    limit = Math.max(0, required - body.h);
    scroll = Math.min(scroll, limit);
    text(
      'DIAGNOSTICS · H CLOSE' + (limit ? ' · SCROLL' : ''),
      heading,
      font('sans_bold', 12 * s),
      MUTED,
    );
    ctx.beginPath();
    ctx.rect(body.x, body.y, body.w, body.h);
    ctx.clip();
    const chart = rect(inner.x, body.y - scroll, sectionW, chartH),
      rays = rect(chart.x, chart.y + chart.h + Math.trunc(20 * s), sectionW, raysH);
    const controls = horizontal
      ? rect(
          chart.x + chart.w + 2 * gap,
          chart.y,
          inner.x + inner.w - chart.x - chart.w - 2 * gap,
          controlsH,
        )
      : rect(inner.x, rays.y + rays.h + Math.trunc(24 * s), inner.w, controlsH);
    text('STEERING', rect(chart.x, chart.y, chart.w, 22 * s), arcade(s), AMBER);
    const labelPad = Math.floor(assets.measure('-35°', sm(s)) / 2) + 2,
      plot = rect(
        chart.x + labelPad,
        chart.y + Math.trunc(34 * s),
        Math.max(1, chart.w - 2 * labelPad),
        64 * s,
      );
    const binX = (angle: number) =>
      plot.x + plot.w / 30 + ((angle + 35) / 70) * ((plot.w * 14) / 15);
    if (game.mode === 'AUTONOMOUS') {
      const max = Math.max(0.2, ...game.probabilities),
        best = game.probabilities.indexOf(Math.max(...game.probabilities));
      game.probabilities.forEach((p, i) => {
        const x0 = plot.x + Math.round((i * plot.w) / 15),
          x1 = plot.x + Math.round(((i + 1) * plot.w) / 15) - 2,
          h = Math.max(2, Math.round((p / max) * plot.h));
        fill(
          ctx,
          rect(x0, plot.y + plot.h - h, Math.max(1, x1 - x0), h),
          i === best ? AMBER : [72, 77, 77],
        );
      });
      const angle = game.probabilities.reduce((sum, p, i) => sum + p * (-35 + 5 * i), 0),
        x = Math.round(binX(angle)),
        radius = Math.max(3, Math.round(4 * s));
      line(
        ctx,
        [x, plot.y - 3],
        [x, plot.y + plot.h + 2],
        GREEN,
        Math.max(1, 2 * Math.round(s) - 1),
      );
      polygon(
        ctx,
        [
          [x - radius, plot.y - 8],
          [x + radius, plot.y - 8],
          [x, plot.y - 3],
        ],
        GREEN,
      );
    } else {
      const angle = Math.max(-35, Math.min(35, (game.state.phi * 180) / Math.PI)),
        bar = rect(plot.x, plot.y + Math.floor(plot.h / 2) - Math.trunc(6 * s), plot.w, 12 * s),
        cursor = Math.round(plot.x + ((angle + 35) / 70) * plot.w),
        middle = plot.x + Math.floor(plot.w / 2);
      fill(ctx, bar, [53, 56, 56]);
      fill(
        ctx,
        rect(Math.min(middle, cursor), bar.y, Math.max(1, Math.abs(cursor - middle)), bar.h),
        AMBER,
      );
      line(ctx, [middle, bar.y - 4], [middle, bar.y + bar.h + 4], TEXT);
      line(ctx, [cursor, bar.y - 5], [cursor, bar.y + bar.h + 5], AMBER, 2);
    }
    for (const angle of [-35, 0, 35]) {
      const x = Math.round(
        game.mode === 'AUTONOMOUS' ? binX(angle) : plot.x + ((angle + 35) / 70) * plot.w,
      );
      line(ctx, [x, plot.y + plot.h + 2], [x, plot.y + plot.h + 6], MUTED);
      const label = assets.glyph(angle ? `${angle > 0 ? '+' : ''}${angle}°` : '0°', sm(s), MUTED);
      ctx.drawImage(label, x - Math.floor(label.width / 2), plot.y + plot.h + Math.trunc(10 * s));
    }
    const toggleW = assets.measure('OFF', arcade(s)) + assets.measure('V', sm(s)) + 4 * gap,
      toggle = rect(rays.x + rays.w - toggleW, rays.y, toggleW, 30 * s);
    text(
      'ATTENTION RAYS',
      rect(rays.x, rays.y, rays.w - toggleW - gap, toggle.h),
      arcade(s),
      AMBER,
    );
    button('rays', toggle, game.rays ? 'ON' : 'OFF', 'V', game.rays);
    lines.forEach((value, i) =>
      text(
        value,
        rect(rays.x, rays.y + Math.trunc((44 + 16 * i) * s), rays.w, 16 * s),
        sm(s),
        MUTED,
      ),
    );
    for (const section of [rays, controls])
      if (section.y > chart.y)
        line(
          ctx,
          [section.x, section.y - Math.trunc(10 * s)],
          [section.x + section.w, section.y - Math.trunc(10 * s)],
          BORDER,
        );
    text('CONTROLS', rect(controls.x, controls.y, controls.w, 24 * s), arcade(s), AMBER);
    const keys = [
      ['W/S', 'Gas / reverse'],
      ['A/D', 'Steer'],
      ['M', 'Auto / manual'],
      ['T', '1x / 2x / 4x / 8x / 16x'],
      ['Space', 'Pause'],
      ['R', 'Reset'],
      ['E', 'Draw path'],
      ['V', 'Attention rays'],
      ['Esc', 'Cancel drawing'],
    ];
    const rowH = Math.max(
      14,
      Math.min(Math.trunc(25 * s), Math.floor((controls.h - Math.trunc(32 * s)) / keys.length)),
    );
    keys.forEach(([key, label], i) => {
      const row = rect(controls.x, controls.y + Math.trunc(32 * s) + i * rowH, controls.w, rowH),
        kw = Math.min(Math.trunc(55 * s), Math.floor(row.w / 3));
      text(key, rect(row.x, row.y, kw, row.h), sm(s), AMBER);
      text(label, rect(row.x + kw, row.y, Math.max(1, row.w - kw), row.h), sm(s), MUTED);
    });
    ctx.restore();
  };
  const drawAlert = (game: Game) => {
    const { scale: s, pad, gap } = current;
    const title = game.failure ? 'CONTROLLER ERROR' : game.status || (game.paused ? 'PAUSED' : '');
    get('alert').hidden = !title || game.drawing;
    get('alert').querySelector('strong')!.textContent = title;
    if (!title || game.drawing) return;
    const accent =
      game.status === 'DOCK SECURED' ? GREEN : game.status || game.failure ? RED : AMBER;
    const subtitle =
      game.failure ||
      {
        JACKKNIFED: 'HITCH LIMIT REACHED',
        'OFF TRACK': 'RETRY OR DRAW A WIDER ROUTE',
        'DOCK SECURED': 'DELIVERY COMPLETE',
        '': 'SPACE TO CONTINUE',
      }[game.status];
    let view = current.yard;
    if (view.h < Math.trunc(170 * s) || view.w < Math.trunc(280 * s))
      view = rect(0, current.toolbar.h, current.width, current.footer.y - current.toolbar.h);
    const w = Math.max(1, Math.min(Math.trunc(520 * s), view.w - 2 * pad)),
      h = Math.max(1, Math.min(Math.trunc(172 * s), view.h - 2 * gap));
    const r = rect(
      view.x + Math.floor(view.w / 2) - Math.floor(w / 2),
      view.y + Math.floor(view.h / 2) - Math.floor(h / 2),
      w,
      h,
    );
    polygon(ctx, corners(move(r, 4, 5), 8), [8, 9, 10]);
    polygon(ctx, corners(r, 8), PANEL);
    polygon(ctx, corners(r, 8), accent, 2);
    let key = font('arcade', 24 * s);
    if (assets.measure(title, key) > r.w - 2 * pad) key = font('arcade', 18 * s);
    if (assets.measure(title, key) > r.w - 2 * pad) key = arcade(s);
    const compact = r.h < Math.trunc(156 * s),
      tr = rect(r.x + pad, r.y + Math.trunc((compact ? 12 : 24) * s), r.w - 2 * pad, 34 * s);
    text(title, move(tr, 2, 2), key, [65, 53, 32], true);
    text(title, tr, key, accent, true);
    if (!compact)
      text(
        subtitle,
        rect(r.x + pad, tr.y + tr.h + Math.trunc(8 * s), r.w - 2 * pad, 22 * s),
        font('mono', 13 * s),
        TEXT,
        true,
      );
    const bw = Math.min(Math.trunc(180 * s), r.w - 2 * pad),
      bh = Math.trunc(34 * s),
      retry = Boolean(game.status || game.failure);
    button(
      'alert-reset',
      rect(
        r.x + Math.floor(r.w / 2) - Math.floor(bw / 2),
        r.y + r.h - Math.trunc(16 * s) - bh,
        bw,
        bh,
      ),
      retry ? 'RETRY' : 'CONTINUE',
      retry ? 'R' : 'Space',
      false,
      true,
    );
  };
  return {
    get,
    canvas,
    setAssets(value: Assets) {
      assets = value;
    },
    loading(message: string) {
      get('loading').querySelector('p')!.textContent = message;
    },
    ready() {
      get('loading').hidden = true;
    },
    failure(message: string) {
      get('loading').hidden = false;
      get('loading').querySelector('strong')!.textContent = 'UNABLE TO START';
      get('loading').querySelector('p')!.textContent = message;
    },
    getLayout() {
      return current;
    },
    update(game: Game) {
      if (!assets) return;
      const { top, bottom } = layout(game),
        { width, height, scale: s, pad, gap, toolbar, footer, yard } = current;
      if (overlay.width !== width || overlay.height !== height) {
        overlay.width = width;
        overlay.height = height;
        overlay.style.width = `${width}px`;
        overlay.style.height = `${height}px`;
      }
      ctx.clearRect(0, 0, width, height);
      ctx.imageSmoothingEnabled = false;
      specs = [];
      Object.assign(canvas.style, {
        left: `${yard.x}px`,
        top: `${yard.y}px`,
        width: `${yard.w}px`,
        height: `${yard.h}px`,
      });
      panel(toolbar);
      line(ctx, [0, toolbar.h], [width, toolbar.h], BORDER, 2);
      if (game.drawing) {
        const cancel = rect(width - pad - Math.trunc(96 * s), pad, 96 * s, 40 * s);
        text(
          'DRAW ROUTE',
          rect(pad, pad, Math.max(1, cancel.x - 2 * pad), 40 * s),
          arcade(s),
          AMBER,
        );
        text(
          'Drag · Release to finish · Esc cancel',
          rect(pad, toolbar.h - Math.trunc(25 * s), width - 2 * pad, 20 * s),
          sm(s),
          MUTED,
        );
        button('cancel', cancel, 'CANCEL', '', false, true);
      } else {
        const title = rect(top[0].x, top[0].y, top[0].w, 25 * s);
        text('JACKKNIFE', move(title, 1, 2), font('arcade', 18 * s), [87, 67, 35]);
        text('JACKKNIFE', title, font('arcade', 18 * s), AMBER);
        text(
          'B-DOUBLE // REVERSE',
          rect(title.x, title.y + title.h, top[0].w, 15 * s),
          sm(s),
          MUTED,
        );
        const labels = [
          ['mode', game.mode === 'AUTONOMOUS' ? 'AUTO' : 'MANUAL', 'M'],
          ['draw', 'DRAW', 'E'],
          ['new', 'NEW ROUTE', 'P'],
          ['speed', `${game.speedMultiplier}X`, 'T'],
          ['pause', game.paused ? 'CONTINUE' : 'PAUSE', 'Space'],
          ['telemetry', 'TELEMETRY', 'H'],
        ];
        labels.forEach(([id, label, key], i) =>
          button(
            id,
            top[i + 1],
            label,
            key,
            (id === 'telemetry' && game.diagnostics) ||
              (id === 'mode' && game.mode === 'AUTONOMOUS'),
            id === 'draw',
          ),
        );
      }
      panel(footer);
      line(ctx, [0, footer.y], [width, footer.y], BORDER, 2);
      let status = game.drawing
        ? 'DRAW ROUTE'
        : game.status === 'JACKKNIFED'
          ? 'JACKKNIFED'
          : game.status === 'DOCK SECURED'
            ? 'COMPLETE'
            : game.paused
              ? 'PAUSED'
              : game.mode;
      text(
        status,
        bottom[0],
        arcade(s),
        game.status === 'JACKKNIFED' ? RED : game.status === 'DOCK SECURED' ? GREEN : AMBER,
      );
      const stability = bottom[1],
        lw = assets.measure('STABILITY', small(s));
      text('STABILITY', rect(stability.x, stability.y, lw, stability.h), small(s), MUTED);
      const bar = rect(
        stability.x + lw + gap,
        stability.y + Math.floor(stability.h / 2) - Math.trunc(4 * s),
        112 * s,
        8 * s,
      );
      for (let i = 0; i < 14; i++) {
        const left = bar.x + Math.round((i * bar.w) / 14),
          right = bar.x + Math.round(((i + 1) * bar.w) / 14) - Math.max(1, Math.trunc(2 * s));
        fill(
          ctx,
          rect(left, bar.y, Math.max(1, right - left), bar.h),
          i / 14 < game.stability
            ? game.stability > 0.45
              ? GREEN
              : game.stability > 0.2
                ? AMBER
                : RED
            : [53, 56, 56],
        );
      }
      const infer = bottom[2],
        ilw = assets.measure('INFERENCE', small(s));
      text('INFERENCE', rect(infer.x, infer.y, ilw, infer.h), small(s), MUTED);
      text(
        `${game.inferenceMs.toFixed(1)} ms`,
        rect(infer.x + ilw + gap, infer.y, infer.w - ilw - gap, infer.h),
        font('mono', 13 * s),
        AMBER,
      );
      button('reset', bottom[3], 'RESET', 'R');
      if (current.drawer.w) diagnostics(game);
      drawAlert(game);
      get('drawer').hidden = !game.diagnostics;
      get('policy-panel').hidden = game.mode !== 'AUTONOMOUS';
      get('drawing-hint').hidden = !game.drawing;
      for (const child of get('hit-targets').querySelectorAll<HTMLButtonElement>('button'))
        child.hidden = true;
      for (const spec of specs) {
        const node = get<HTMLButtonElement>(spec.id);
        node.hidden = false;
        node.textContent = spec.label;
        node.setAttribute(
          'aria-label',
          spec.id === 'speed' ? `${game.speedMultiplier} times speed` : spec.label,
        );
        node.setAttribute('aria-pressed', String(Boolean(spec.active)));
        Object.assign(node.style, {
          left: `${spec.r.x}px`,
          top: `${spec.r.y}px`,
          width: `${spec.r.w}px`,
          height: `${spec.r.h}px`,
        });
      }
      get('telemetry').setAttribute('aria-expanded', String(game.diagnostics));
    },
  };
}
