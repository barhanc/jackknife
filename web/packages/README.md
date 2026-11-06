# Local ruNNtime package

`jackknife-runntime-core-0.2.0-f6db0ea.tgz` is installed as a normal dependency:

```json
"@jackknife/runntime-core": "file:packages/jackknife-runntime-core-0.2.0-f6db0ea.tgz"
```

The tarball contains bundled ESM, TypeScript declarations, the upstream MIT
license, and source provenance. GPU kernels are precompiled by the TypeGPU build
plugin. The game imports `@jackknife/runntime-core`; no upstream source checkout
is needed for development, CI, or deployment.

Source: https://github.com/software-mansion/runntime

Revision: `f6db0ea8937f9208c28b8ea69a5fdd8a0bf4320c`

Directory: `packages/runntime/src/core` (unmodified).

The published upstream package exposes its model zoo, but not the core API needed
for this custom policy. This private package exposes that core without changing
its implementation. Its name is local to this repository; it is not an official
ruNNtime release.

To rebuild using the tools pinned in package-lock.json (Node 22.18+ or 24.11+ LTS):

```sh
npm ci
npm run pack:runntime
npm install ./packages/jackknife-runntime-core-0.2.0-f6db0ea.tgz
npm run build
npm run test:browser
```

The rebuild script clones the pinned revision into a temporary directory, bundles
the core with its types and compiled GPU kernels, runs npm pack, then removes the
temporary sources. Commit the tarball and updated lockfile together. Updating the
revision also requires updating the package version in the script and checking
steering/attention parity against PyTorch.
