![Manual steering at 2× and jackknife recovery, complex route drawing, and autonomous reversing at 16× with telemetry and attention rays](.github/assets/showcase.gif)

## Quick Start

This repository uses [`uv`](https://docs.astral.sh/uv/) for fast, reproducible
dependency management and execution.

[Install uv](https://docs.astral.sh/uv/getting-started/installation/).

Install Dependencies

```bash
uv sync
```

Run the simulation in autonomous mode (loads `models/model.pt` by default):

```bash
uv run python main.py
```

Run tests:

```bash
uv run python -m unittest discover -s tests -v
```

Train the model:

```bash
uv run python main.py --train --epochs 4 --batch-size 64 --lr 3e-4 --stream --trajectories 60
```
