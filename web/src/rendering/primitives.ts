/** Batched alpha-blended polygon rendering through TypeGPU. */
import tgpu, { d, type TgpuRoot } from 'typegpu';
import type { Point } from '../simulation/physics';
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
    readonly scaleX: number,
    readonly scaleY: number,
    readonly width: number,
    readonly height: number,
  ) {}
  vertex(p: Point, color: Color) {
    const pixelX = Math.trunc(Math.floor(this.width / 2) + (p[0] * this.scaleX * this.width) / 2);
    const pixelY = Math.trunc(Math.floor(this.height / 2) - (p[1] * this.scaleY * this.height) / 2);
    this.data.push(
      (pixelX / this.width) * 2 - 1,
      1 - (pixelY / this.height) * 2,
      0,
      0,
      Math.round(color[0]) / 255,
      Math.round(color[1]) / 255,
      Math.round(color[2]) / 255,
      color[3] ?? 1,
    );
  }
  polygon(points: Point[], color: Color) {
    for (let i = 1; i < points.length - 1; i++)
      for (const p of [points[0], points[i], points[i + 1]]) this.vertex(p, color);
  }
  line(a: Point, b: Point, width: number, color: Color) {
    width = Math.max(width, 2 / (this.scaleX * this.width));
    const length = Math.hypot(b[0] - a[0], b[1] - a[1]);
    if (length < 1e-9) return;
    const nx = ((-(b[1] - a[1]) / length) * width) / 2,
      ny = (((b[0] - a[0]) / length) * width) / 2;
    this.polygon(
      [
        [a[0] + nx, a[1] + ny],
        [a[0] - nx, a[1] - ny],
        [b[0] - nx, b[1] - ny],
        [b[0] + nx, b[1] + ny],
      ],
      color,
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
