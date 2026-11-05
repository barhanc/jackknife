import os
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

from src.kinematics import TractorDoubleTrailerKinematics, VehicleDimensions, VehicleState
from src.path_generator import Path
from src.model import MultiTrailerPFN

def create_complex_benchmark_path() -> Path:
    """
    Creates a fixed challenging benchmark trajectory:
    An 80-meter path consisting of:
    1. Straight lead-in (15m)
    2. Sharp counter-curving chicane (35m)
    3. Curved docking bay entry (30m)
    """
    s = np.linspace(0.0, 75.0, 150)
    # Target path: Trailer 2 reverses from (0,0) along negative X with lateral S-curve
    x = -s
    # Double S-curve with increasing difficulty
    y = 5.0 * np.sin(s * 0.11) + 2.0 * np.sin(s * 0.05)
    return Path(x, y, ds=0.8)

def rollout_policy(model: MultiTrailerPFN, path: Path, device: torch.device, max_steps: int = 500, dt: float = 0.05):
    dims = VehicleDimensions()
    kinematics = TractorDoubleTrailerKinematics(dims)

    # Initial state: Trailer 2 aligned at path start
    th_path = path.headings[0]
    th2 = th_path - math.pi
    x2, y2 = path.x[0], path.y[0]
    th1 = th2
    th0 = th2
    x1 = x2 + dims.L2 * math.cos(th1)
    y1 = y2 + dims.L2 * math.sin(th1)
    x0 = x1 + dims.L1 * math.cos(th0)
    y0 = y1 + dims.L1 * math.sin(th0)

    state = VehicleState(float(x0), float(y0), float(th0), 0.0, 0.0, 0.0)

    history = {
        'time': [],
        't2_x': [], 't2_y': [], 't2_th': [],
        't1_x': [], 't1_y': [],
        'cab_x': [], 'cab_y': [],
        'beta1': [], 'beta2': [],
        'phi': [],
        'cross_track_err': [],
        'rho': [],
        'jackknifed': False,
        'jackknife_step': None
    }

    target_steer = 0.0

    for step in range(max_steps):
        t = step * dt
        x2_cur, y2_cur, th2_cur = kinematics.get_trailer2_axle(state)
        x1_cur, y1_cur, _ = kinematics.get_trailer1_axle(state)

        # Cross track error
        nearest_idx = path.find_nearest_index(x2_cur, y2_cur)
        px, py = path.x[nearest_idx], path.y[nearest_idx]
        e_y = math.hypot(x2_cur - px, y2_cur - py)

        rho = kinematics.jackknife_proximity(state)

        history['time'].append(t)
        history['t2_x'].append(x2_cur)
        history['t2_y'].append(y2_cur)
        history['t2_th'].append(th2_cur)
        history['t1_x'].append(x1_cur)
        history['t1_y'].append(y1_cur)
        history['cab_x'].append(state.x0)
        history['cab_y'].append(state.y0)
        history['beta1'].append(math.degrees(state.beta1))
        history['beta2'].append(math.degrees(state.beta2))
        history['phi'].append(math.degrees(state.phi))
        history['cross_track_err'].append(e_y)
        history['rho'].append(rho)

        if kinematics.is_jackknifed(state):
            history['jackknifed'] = True
            history['jackknife_step'] = step
            break

        # Stop if reached end of path
        if nearest_idx >= len(path.x) - 3:
            break

        # Inference
        tokens = path.get_lookahead_tokens(x2_cur, y2_cur, th2_cur, num_tokens=64, ds=0.8)
        hitch_state = np.array([
            math.sin(state.beta1), math.cos(state.beta1),
            math.sin(state.beta2), math.cos(state.beta2)
        ], dtype=np.float32)

        steer_cmd, _, _ = model.predict_action(tokens, hitch_state, device)
        target_steer = steer_cmd

        # Step kinematics
        state = kinematics.step_rk4(state, v0=-1.5, target_phi=target_steer, dt=dt)

    return history


def run_progression_analysis(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = create_complex_benchmark_path()

    stages = [
        ("Epoch 0 (Untrained)", "models/checkpoint_epoch0.pt", "#e74c3c", "--"),
        ("Epoch 1 (Initial Learning)", "models/checkpoint_epoch1.pt", "#e67e22", "-."),
        ("Epoch 5 (Mid Training)", "models/checkpoint_epoch5.pt", "#f1c40f", ":"),
        ("Epoch 20 (Fully Trained PFN)", "models/pfn_best.pt", "#2ecc71", "-")
    ]

    results = []

    for label, ckpt_path, color, style in stages:
        if not os.path.exists(ckpt_path):
            print(f"Skipping {label}, {ckpt_path} not found.")
            continue
        model = MultiTrailerPFN().to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()

        print(f"Evaluating {label}...")
        hist = rollout_policy(model, path, device)
        results.append((label, hist, color, style))

    # Plot Multi-Panel Figure
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(18, 12), dpi=150)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.3, 1, 1], hspace=0.35, wspace=0.25)

    # 1. Spatial Trajectory Map (Top view)
    ax_map = fig.add_subplot(gs[0, :])
    ax_map.set_title("Fixed Complex Benchmark: Closed-Loop Reversing Trajectory Comparison", fontsize=14, fontweight='bold', pad=12, color='#ffffff')
    ax_map.plot(path.x, path.y, color='#5dade2', linewidth=3, label="Reference Target Path (Trailer 2 Axle)", alpha=0.9)

    # Corridor boundaries
    for side in [-3.5, 3.5]:
        corridor_x = path.x - side * np.sin(path.headings)
        corridor_y = path.y + side * np.cos(path.headings)
        ax_map.plot(corridor_x, corridor_y, color='#34495e', linestyle='--', linewidth=1.2, alpha=0.7)

    for label, hist, color, style in results:
        ax_map.plot(hist['t2_x'], hist['t2_y'], color=color, linestyle=style, linewidth=2.5, label=f"{label} (T2 Axle)")
        if hist['jackknifed']:
            ax_map.scatter([hist['t2_x'][-1]], [hist['t2_y'][-1]], color='#ff0000', marker='X', s=140, zorder=5, label=f"{label} Jackknife Lockup")

    # Show Cab counter-steering trajectory for Epoch 20
    for label, hist, color, style in results:
        if "Epoch 20" in label:
            ax_map.plot(hist['cab_x'], hist['cab_y'], color='#f39c12', linestyle=':', linewidth=1.5, alpha=0.75, label="Cab Steering Path (Epoch 20)")

    ax_map.set_xlabel("Global X (meters)", fontsize=11, color='#bdc3c7')
    ax_map.set_ylabel("Global Y (meters)", fontsize=11, color='#bdc3c7')
    ax_map.grid(True, linestyle=':', alpha=0.3)
    ax_map.legend(loc='upper right', fontsize=10, framealpha=0.85)

    # 2. Cross-Track Error
    ax_err = fig.add_subplot(gs[1, 0])
    ax_err.set_title("Cross-Track Lateral Tracking Error $e_y$", fontsize=12, fontweight='bold', color='#ffffff')
    for label, hist, color, style in results:
        ax_err.plot(hist['time'], hist['cross_track_err'], color=color, linestyle=style, linewidth=2, label=label)
    ax_err.set_xlabel("Simulation Time (s)", fontsize=10, color='#bdc3c7')
    ax_err.set_ylabel("Error (meters)", fontsize=10, color='#bdc3c7')
    ax_err.grid(True, linestyle=':', alpha=0.3)
    ax_err.legend(fontsize=9, loc='upper left')

    # 3. Steering Command phi(t)
    ax_steer = fig.add_subplot(gs[1, 1])
    ax_steer.set_title("Steering Wheel Angle $\phi(t)$ (Non-Minimum Phase Counter-Steering)", fontsize=12, fontweight='bold', color='#ffffff')
    ax_steer.axhline(35.0, color='#e74c3c', linestyle=':', alpha=0.5, label="Mechanical Limit (±35°)")
    ax_steer.axhline(-35.0, color='#e74c3c', linestyle=':', alpha=0.5)
    for label, hist, color, style in results:
        ax_steer.plot(hist['time'], hist['phi'], color=color, linestyle=style, linewidth=1.8, label=label)
    ax_steer.set_xlabel("Simulation Time (s)", fontsize=10, color='#bdc3c7')
    ax_steer.set_ylabel("Steer Angle (°)", fontsize=10, color='#bdc3c7')
    ax_steer.grid(True, linestyle=':', alpha=0.3)
    ax_steer.legend(fontsize=9, loc='lower left')

    # 4. Hitch Angles beta1 & beta2
    ax_hitch = fig.add_subplot(gs[2, 0])
    ax_hitch.set_title("Hitch Articulation Angles $\\beta_1$ & $\\beta_2$ vs Jackknife Boundary", fontsize=12, fontweight='bold', color='#ffffff')
    ax_hitch.axhspan(70.0, 90.0, color='#e74c3c', alpha=0.2, label="Critical Jackknife Zone (≥70°)")
    ax_hitch.axhspan(-90.0, -70.0, color='#e74c3c', alpha=0.2)
    ax_hitch.axhline(70.0, color='#e74c3c', linestyle='--', alpha=0.6)
    ax_hitch.axhline(-70.0, color='#e74c3c', linestyle='--', alpha=0.6)

    # Plot Epoch 1 vs Epoch 20 hitch angles
    for label, hist, color, style in results:
        if "Epoch 0" in label:
            ax_hitch.plot(hist['time'], hist['beta1'], color=color, linestyle=style, alpha=0.7, label=f"{label} $\\beta_1$")
        elif "Epoch 20" in label:
            ax_hitch.plot(hist['time'], hist['beta1'], color='#2ecc71', linestyle='-', linewidth=2, label="Epoch 20 $\\beta_1$ (Cab-T1)")
            ax_hitch.plot(hist['time'], hist['beta2'], color='#3498db', linestyle='-', linewidth=2, label="Epoch 20 $\\beta_2$ (T1-T2)")

    ax_hitch.set_xlabel("Simulation Time (s)", fontsize=10, color='#bdc3c7')
    ax_hitch.set_ylabel("Hitch Angle (°)", fontsize=10, color='#bdc3c7')
    ax_hitch.set_ylim(-90, 90)
    ax_hitch.grid(True, linestyle=':', alpha=0.3)
    ax_hitch.legend(fontsize=9, loc='lower right')

    # 5. Stability Proximity Margin
    ax_rho = fig.add_subplot(gs[2, 1])
    ax_rho.set_title("Stability Margin $\\rho(t) = 1 - \\max(|\\beta_i| / 70^\\circ)$", fontsize=12, fontweight='bold', color='#ffffff')
    for label, hist, color, style in results:
        ax_rho.plot(hist['time'], np.array(hist['rho']) * 100.0, color=color, linestyle=style, linewidth=2, label=label)
    ax_rho.axhline(0.0, color='#e74c3c', linestyle='--', alpha=0.6, label="Jackknife Boundary (0%)")
    ax_rho.set_xlabel("Simulation Time (s)", fontsize=10, color='#bdc3c7')
    ax_rho.set_ylabel("Margin (%)", fontsize=10, color='#bdc3c7')
    ax_rho.set_ylim(-5, 105)
    ax_rho.grid(True, linestyle=':', alpha=0.3)
    ax_rho.legend(fontsize=9, loc='upper right')

    fig.tight_layout()
    out_img = os.path.join(output_dir, "controller_training_progression.png")
    fig.savefig(out_img, bbox_inches='tight')
    plt.close(fig)
    print(f"Progression figure saved to {out_img}")
    return out_img

if __name__ == "__main__":
    import sys
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "artifacts"
    run_progression_analysis(out_dir)
