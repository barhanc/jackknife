/** Rebuild the local ruNNtime core package from an exact upstream revision. */
import { execFileSync } from 'node:child_process';
import { cp, mkdir, mkdtemp, rm, symlink, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { build } from 'tsdown';
import typegpu from 'unplugin-typegpu/rollup';

const revision = 'f6db0ea8937f9208c28b8ea69a5fdd8a0bf4320c';
const web = dirname(dirname(fileURLToPath(import.meta.url)));
const temporary = await mkdtemp(join(tmpdir(), 'jackknife-runntime-package-'));
const checkout = join(temporary, 'upstream');
const packageDir = join(temporary, 'package');
const destination = join(web, 'packages');
const run = (command, args, cwd) => execFileSync(command, args, { cwd, stdio: 'inherit' });
try {
  run(
    'git',
    [
      'clone',
      '--quiet',
      '--no-checkout',
      '--filter=blob:none',
      'https://github.com/software-mansion/runntime.git',
      checkout,
    ],
    web,
  );
  run('git', ['checkout', '--quiet', '--detach', revision], checkout);
  await mkdir(join(packageDir, 'src'), { recursive: true });
  await cp(join(checkout, 'packages/runntime/src/core'), join(packageDir, 'src/core'), {
    recursive: true,
  });
  await cp(join(checkout, 'LICENSE.md'), join(packageDir, 'LICENSE.md'));
  await symlink(join(web, 'node_modules'), join(packageDir, 'node_modules'), 'dir');
  await writeFile(
    join(packageDir, 'package.json'),
    JSON.stringify(
      {
        name: '@jackknife/runntime-core',
        version: '0.2.0-f6db0ea',
        private: true,
        type: 'module',
        license: 'MIT',
        description:
          'Pinned ruNNtime core for the Jackknife browser game; not an official upstream package.',
        repository: 'https://github.com/software-mansion/runntime',
        files: ['dist', 'LICENSE.md', 'UPSTREAM.md'],
        exports: { '.': { types: './dist/index.d.mts', default: './dist/index.mjs' } },
        peerDependencies: { typegpu: '0.12.5' },
      },
      null,
      2,
    ),
  );
  await writeFile(
    join(packageDir, 'UPSTREAM.md'),
    `Source: https://github.com/software-mansion/runntime\nRevision: ${revision}\nDirectory: packages/runntime/src/core\nUnmodified source, bundled with TypeGPU kernels compiled at package-build time.\n`,
  );
  await writeFile(
    join(packageDir, 'tsconfig.json'),
    JSON.stringify({
      compilerOptions: {
        target: 'ES2022',
        module: 'ESNext',
        moduleResolution: 'Bundler',
        strict: true,
        skipLibCheck: true,
        declaration: true,
        allowImportingTsExtensions: true,
        lib: ['tsover', 'ES2022', 'DOM'],
        types: ['@webgpu/types'],
      },
      include: ['src'],
    }),
  );
  await build({
    cwd: packageDir,
    config: false,
    entry: { index: 'src/core/index.ts' },
    outDir: 'dist',
    format: 'esm',
    dts: true,
    sourcemap: false,
    plugins: [typegpu({ exclude: ['**/*.d.ts', '**/*.d.mts'] })],
    deps: { neverBundle: ['typegpu'] },
    clean: true,
    tsconfig: join(packageDir, 'tsconfig.json'),
  });
  await mkdir(destination, { recursive: true });
  run('npm', ['pack', '--ignore-scripts', '--pack-destination', destination], packageDir);
} finally {
  await rm(temporary, { recursive: true, force: true });
}
