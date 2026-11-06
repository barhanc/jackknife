/** Batched alpha-blended polygon rendering through TypeGPU. */
import tgpu, { d, type TgpuRoot } from 'typegpu';
import type { Point } from '../simulation/physics';
import { fillPolygon, strokeLine } from './pixelRaster';
export type Color = [number, number, number, number?];
const Vertex = d.struct({ position: d.vec2f, color: d.vec4f });
const layout = tgpu.vertexLayout((n) => d.arrayOf(Vertex, n));
const vertex = tgpu.vertexFn({
  in: { position: d.vec2f, color: d.vec4f },
  out: { position: d.builtin.position, color: d.vec4f },
})(({ position, color }) => {
  'use gpu';
  return { position: d.vec4f(position, 0, 1), color };
});
const fragment = tgpu.fragmentFn({ in: { color: d.vec4f }, out: d.vec4f })(({ color }) => {
  'use gpu';
  return color;
});
export class Mesh {
  data: number[] = [];
  constructor(
    readonly scale: number,
    readonly width: number,
    readonly height: number,
  ) {}
  project(p: Point): Point {
    return [
      Math.trunc(Math.floor(this.width / 2) + p[0] * this.scale),
      Math.trunc(Math.floor(this.height / 2) - p[1] * this.scale),
    ];
  }
  private span(y: number, left: number, right: number, color: Color) {
    if (y < 0 || y >= this.height) return;
    left = Math.max(0, left);
    right = Math.min(this.width - 1, right);
    if (right < left) return;
    const x0 = (left / this.width) * 2 - 1,
      x1 = ((right + 1) / this.width) * 2 - 1;
    const y0 = 1 - (y / this.height) * 2,
      y1 = 1 - ((y + 1) / this.height) * 2;
    const r = color[0] / 255,
      g = color[1] / 255,
      b = color[2] / 255,
      a = color[3] ?? 1;
    this.data.push(
      x0,
      y0,
      0,
      0,
      r,
      g,
      b,
      a,
      x1,
      y0,
      0,
      0,
      r,
      g,
      b,
      a,
      x1,
      y1,
      0,
      0,
      r,
      g,
      b,
      a,
      x0,
      y0,
      0,
      0,
      r,
      g,
      b,
      a,
      x1,
      y1,
      0,
      0,
      r,
      g,
      b,
      a,
      x0,
      y1,
      0,
      0,
      r,
      g,
      b,
      a,
    );
  }
  polygon(points: Point[], color: Color) {
    fillPolygon(
      points.map((p) => this.project(p)),
      (y, left, right) => this.span(y, left, right, color),
    );
  }
  /** Stroke width is measured in screen pixels, as in pygame.draw.lines. */
  line(a: Point, b: Point, width: number, color: Color) {
    strokeLine(this.project(a), this.project(b), width, (y, left, right) =>
      this.span(y, left, right, color),
    );
  }
}
export function createPainter(root: TgpuRoot, canvas: HTMLCanvasElement) {
  const context = root.configureContext({ canvas, alphaMode: 'premultiplied' });
  const pipeline = root.createRenderPipeline({
    vertex,
    fragment,
    attribs: layout.attrib,
    targets: {
      blend: {
        color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
        alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
      },
    },
  });
  let dynamicBuffer = root.device.createBuffer({
    size: 1024 * 1024,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  return {
    draw(mesh: Mesh) {
      const data = new Float32Array(mesh.data);
      if (data.byteLength > dynamicBuffer.size) {
        dynamicBuffer.destroy();
        dynamicBuffer = root.device.createBuffer({
          size: data.byteLength * 2,
          usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        });
      }
      root.device.queue.writeBuffer(dynamicBuffer, 0, data);
      const encoder = root.device.createCommandEncoder();
      const pass = encoder.beginRenderPass({
        colorAttachments: [
          {
            view: context.getCurrentTexture().createView(),
            clearValue: [0, 0, 0, 0],
            loadOp: 'clear',
            storeOp: 'store',
          },
        ],
      });
      if (data.length)
        pipeline
          .with(pass)
          .with(layout, dynamicBuffer)
          .draw(data.length / 8);
      pass.end();
      root.device.queue.submit([encoder.finish()]);
    },
    dispose() {
      dynamicBuffer.destroy();
      context.unconfigure();
    },
  };
}
