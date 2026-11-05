# Jackknife: Autonomous Multi-Trailer Reversing with Prior-Data Fitted Networks (PFN)

An interactive, physics-grounded 2.5D simulation and reinforcement/imitation learning environment for autonomous reversing of tractor-double-trailer rigs along complex paths, high-curvature chicanes, and alley-docking gauntlets.

---

## Overview

Reversing multi-trailer articulated vehicles is a notoriously unstable, non-linear non-holonomic control problem where small steering mistakes can rapidly trigger irrecoverable jackknifing. 

This repository implements:
1. **Accurate Articulated Kinematics**: Exact multi-body physics for a tractor with two trailers, modeling hitch offsets, wheelbases, steering limits, and real-time articulation angles ($\beta_1, \beta_2$).
2. **Prior-Data Fitted Network (PFN)**: A Transformer-based architecture featuring trajectory tokenization, self-attention, cross-attention conditioning on the vehicle's instantaneous geometric state, and a multi-class steering policy output.
3. **2.5D Diorama Simulator**: A modern, clean Pygame-based environment with real-time HUD telemetry, stability margin gauges, interactive camera pan/zoom, manual/autonomous policy switching, and preset challenge tracks.
4. **Data Generation & Training Engine**: Synthetic trajectory generation covering arbitrary Dubins paths, multi-turn gauntlets, and high-angle perturbation funnels designed for rapid policy learning and grokking.

---

## Project Structure

```text
jackknife/
├── main.py                  # Unified entrypoint for game and training
├── pyproject.toml           # uv project configuration and dependencies
├── uv.lock                  # Pinned dependency lockfile
├── .gitignore
├── models/                  # Checkpoint storage (.gitkeep)
└── src/
    ├── kinematics.py        # Tractor + double-trailer kinematic equations & state updates
    ├── lqr_controller.py    # Baseline LQR & geometric controllers for trajectory generation
    ├── model.py             # MultiTrailerPFN Transformer policy architecture
    ├── path_generator.py    # Procedural track generation (chicanes, docks, serpentine)
    ├── data_generator.py    # Synthetic trajectory sampler with perturbation funnels
    ├── train.py             # Training pipeline with streaming data generation & metric logging
    ├── game.py              # 2.5D Pygame simulation, rendering, HUD, and interactive controls
    └── visualize_progress.py # Progress visualization utilities
```

---

## Getting Started with `uv`

This repository uses [`uv`](https://docs.astral.sh/uv/) for fast, reproducible dependency management and execution.

### 1. Install `uv`
If you do not have `uv` installed:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Install Dependencies
```bash
uv sync
```

### 3. Launch the Interactive Simulation
Run the simulation in autonomous mode (loads `models/pfn_best.pt` by default):
```bash
uv run python main.py
```

### 4. Train the Model
Train the Transformer policy using the streaming synthetic data pipeline:
```bash
uv run python main.py --train --epochs 100 --batch-size 64 --lr 3e-4 --stream
```

---

## Interactive Controls

| Key | Action |
| :--- | :--- |
| **Space** | Pause / Resume simulation |
| **A** | Toggle Autonomous PFN Mode / Manual Control |
| **R** | Reset vehicle to track starting position |
| **1 - 5** | Load Preset Challenge Tracks (Preset 1: Alley Dock, Preset 5: Gauntlet) |
| **W / S** | Manual Throttle (Reverse / Forward) |
| **Left / Right** | Manual Steer |
| **T** | Toggle Trail Visualization |
| **Mouse Drag** | Pan Camera |
| **Scroll Wheel** | Zoom In / Out |

---

## License

MIT
