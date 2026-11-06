"""Export the existing policy and deterministic parity cases for the browser."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.model import ControlTransformer
from src.kinematics import TractorDoubleTrailerKinematics, VehicleState
from src.path_generator import Path as Route, smooth_and_constrain_user_path, generate_synthetic_path


def export_path_references(output):
    families = ["random_fourier", "yard_grand_tour", "yard_serpentine_slalom", "yard_double_figure8",
                "yard_perimeter_clover", "self_intersecting_figure8", "loop_the_loop", "sharp_switchbacks",
                "cloverleaf_intersect", "serpentine_knot", "dock_with_360_turnaround", "high_curvature_chicane",
                "multi_scale_s", "segmented_dock", "continuous_arc", "figure8_inflection", "compound_kink",
                "piecewise_clothoid", "rapid_chicane", "z_track", "jagged_corners"]
    factory = np.random.default_rng
    cases = []
    for i, family in enumerate(families):
        tape = []
        class RecordedRandom:
            def __init__(self):
                self.rng = factory(7100 + i)
            def uniform(self, low, high, size=None):
                result = self.rng.uniform(low, high, size)
                for value in np.asarray(result).flat:
                    tape.append({'kind': 'uniform', 'value': float(value), 'low': low, 'high': high})
                return result
            def integers(self, low, high):
                result = self.rng.integers(low, high)
                tape.append({'kind': 'integer', 'value': int(result), 'low': low, 'high': high})
                return result
            def random(self):
                result = self.rng.random()
                tape.append({'kind': 'random', 'value': result})
                return result
            def choice(self, population, size=None, replace=True):
                result = self.rng.choice(population, size, replace=replace)
                tape.append({'kind': 'sign' if size is None else 'sample',
                             'value': np.asarray(result).tolist()})
                return result
        np.random.default_rng = RecordedRandom
        try:
            route = generate_synthetic_path(length_m=260 if family.startswith('yard_') else 180,
                                            ds=.8, max_curvature=.11, path_type=family)
        finally:
            np.random.default_rng = factory
        cases.append({'family': family, 'length': 260 if family.startswith('yard_') else 180, 'tape': tape,
                      'points': np.column_stack([route.x, route.y]).tolist(),
                      'headings': route.headings.tolist(), 'curvatures': route.curvatures.tolist()})
    (output / 'path-reference.json').write_text(json.dumps(cases))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'models/model.pt')
    parser.add_argument('--output', type=Path, default=ROOT / 'web/public/models')
    args = parser.parse_args()
    torch.set_num_threads(1)
    model = ControlTransformer().eval()
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
    args.output.mkdir(parents=True, exist_ok=True)
    entries, chunks, offset = {}, [], 0

    def store(name, data):
        nonlocal offset
        data = np.asarray(data, dtype='<f4').copy(order='C')
        entries[name] = {'shape': list(data.shape), 'offset': offset, 'length': data.size}
        chunks.append(data.tobytes())
        offset += data.nbytes

    # Runtime linears store [in, out]. Q/K scaling is folded into weights and biases.
    for name, value in model.state_dict().items():
        data = value.detach().numpy()
        if name.endswith('in_proj_weight') or name.endswith('in_proj_bias'):
            for label, part in zip(('q', 'k', 'v'), np.split(data, 3)):
                scale = 32 ** -.25 if label in ('q', 'k') else 1
                suffix = 'weight' if name.endswith('weight') else 'bias'
                store(name.rsplit('.', 1)[0] + f'.{label}.{suffix}', (part.T if part.ndim == 2 else part) * scale)
        elif name == 'pos_embed':
            store(name, data[0])
        else:
            store(name, data.T if data.ndim == 2 else data)
    binary = b''.join(chunks)
    (args.output / 'weights.bin').write_bytes(binary)
    manifest = {'version': 1, 'sha256': hashlib.sha256(binary).hexdigest(), 'tensors': entries}
    (args.output / 'manifest.json').write_text(json.dumps(manifest))

    # The desktop's four-turn starting route, including its Path-level smoothing.
    ds = .8
    turns = []
    for length, angle in [(16, 0), (22, math.pi/2), (12, 0), (22, -math.pi/2),
                          (14, 0), (22, -math.pi/2), (12, 0), (22, math.pi/2), (18, 0)]:
        n = round(length/ds)
        turns.extend((angle/(n*ds)*2*np.sin(np.pi*np.linspace(0, 1, n))**2).tolist() if angle else [0]*n)
    heading = np.cumsum(np.array(turns)*ds)
    x, y = np.cumsum(np.cos(heading)*ds), np.cumsum(np.sin(heading)*ds)
    route = Route(x-(x.min()+x.max())/2, y-(y.min()+y.max())/2)
    (args.output / 'track.json').write_text(json.dumps({
        'points': np.column_stack([route.x, route.y]).tolist(),
        'headings': route.headings.tolist(), 'curvatures': route.curvatures.tolist(), 'length': route.total_length,
    }))
    kin = TractorDoubleTrailerKinematics()
    cases = []
    for idx, b1, b2 in [(0, 0, 0), (25, .12, -.08), (70, -.25, .17), (120, .4, -.32), (len(route.x)-8, -.1, .2)]:
        x2, y2, th2 = float(route.x[idx]), float(route.y[idx]), float(route.headings[idx]-math.pi)
        tokens = route.get_lookahead_tokens(x2, y2, th2, current_idx=idx)
        hitch = np.array([math.sin(b1), math.cos(b1), math.sin(b2), math.cos(b2)], dtype=np.float32)
        with torch.no_grad():
            logits, attention = model(torch.from_numpy(tokens)[None], torch.from_numpy(hitch)[None])
        cases.append({'index': idx, 'pose': [x2, y2, th2], 'tokens': tokens.tolist(), 'hitch': hitch.tolist(),
                      'logits': logits[0].tolist(), 'attention': attention[0].tolist()})
    initial = VehicleState(3, -2, .7, .2, -.15, .03)
    state = initial
    for _ in range(150):
        state = kin.step_rk4(state, -1.5, -.18, .02)
    raw: list[tuple[float, float]] = [(-40, -20), (-20, -20), (-5, -5), (10, 20), (30, 25), (48, 4), (30, -20)]
    drawn = smooth_and_constrain_user_path(raw, lead_in_m=12)
    fixtures = {'cases': cases, 'physics': {'initial': vars(initial), 'final': vars(state)},
                'drawing': {'raw': raw, 'points': np.column_stack([drawn.x, drawn.y]).tolist(),
                            'headings': drawn.headings.tolist(), 'curvatures': drawn.curvatures.tolist()}}
    (args.output / 'parity.json').write_text(json.dumps(fixtures))
    export_path_references(args.output)
    from src.export_web_assets import export_web_assets, export_visual_references
    export_web_assets(args.output.parent / 'assets')
    export_visual_references(args.output.parent / 'assets')
    print(f'Exported {len(binary)/1024/1024:.2f} MiB and {len(cases)} parity cases to {args.output}')


if __name__ == '__main__':
    main()
