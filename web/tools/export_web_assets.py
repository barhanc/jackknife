"""Bake shared raster assets and reference scenes directly from the desktop renderer."""
import json
import os
from pathlib import Path

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
import pygame

from src.game.depot import DepotTerrain
from src.game.lighting import TruckLighting
from src.game.weather import Snowfall


def export_web_assets(output: Path):
    pygame.init()
    output.mkdir(parents=True, exist_ok=True)
    font_dir = Path(__file__).resolve().parents[2] / 'src/game/assets/fonts'
    families = {'arcade': 'PressStart2P-Regular.ttf', 'mono': 'DejaVuSansMono.ttf',
                'mono_bold': 'DejaVuSansMono-Bold.ttf', 'sans_bold': 'DejaVuSans-Bold.ttf'}
    chars = ''.join(chr(i) for i in range(32, 127)) + '°·×↻−→▾∷'
    fonts, glyphs = {}, []
    x = y = row_h = 0
    for family, filename in families.items():
        for size in range(8, 49):
            font = pygame.font.Font(str(font_dir / filename), size)
            meta = {'height': font.get_height(), 'glyphs': {}}
            for char in chars:
                surface = font.render(char, False, (255, 255, 255))
                w, h = surface.get_size()
                if x + w + 1 > 2048:
                    x, y, row_h = 0, y + row_h + 1, 0
                ink = list(surface.get_bounding_rect())
                meta['glyphs'][char] = {'rect': [x, y, w, h], 'ink': ink, 'advance': font.size(char)[0]}
                glyphs.append((surface, (x, y)))
                x += w + 1
                row_h = max(row_h, h)
            fonts[f'{family}:{size}'] = meta
    atlas = pygame.Surface((2048, y + row_h), pygame.SRCALPHA)
    for surface, point in glyphs:
        atlas.blit(surface, point)
    pygame.image.save(atlas, str(output / 'fonts.png'))
    (output / 'fonts.json').write_text(json.dumps(fonts))

    terrain = DepotTerrain()
    pygame.image.save(terrain.snow.snow, str(output / 'snow.png'))
    pygame.image.save(terrain.snow.shadow, str(output / 'snow-shadow.png'))
    lighting = TruckLighting()
    lighting._prepare(8)
    sprites = {}
    for name, (surface, origin) in lighting._sprites.items():
        pygame.image.save(surface, str(output / f'light-{name}.png'))
        sprites[name] = {'origin': origin, 'size': list(surface.get_size())}
    (output / 'scenery.json').write_text(json.dumps({
        'slabs': terrain.slabs, 'patches': terrain.patches, 'grain': terrain.grain,
        'stains': terrain.stains, 'flakes': Snowfall().flakes, 'lights': sprites,
    }))
    pygame.quit()


def export_visual_references(output: Path):
    """Capture deterministic desktop states for browser screenshot regression tests."""
    import time
    import numpy as np
    from src.game.simulation import JackknifeGame25D
    from src.kinematics import VehicleState
    scenes = []
    configurations = [
        ('manual', 1400, 900, False, False, False),
        ('telemetry', 1400, 900, True, False, False),
        ('drawing', 1400, 900, True, True, False),
        ('autonomous', 1920, 1080, True, False, True),
        ('compact', 800, 700, True, False, False),
    ]
    for name, width, height, diagnostics, drawing, autonomous in configurations:
        game = JackknifeGame25D(None)
        game.width, game.height = width, height
        game.screen = pygame.display.set_mode((width, height))
        game.dynamic_alpha_surf = pygame.Surface((width, height), pygame.SRCALPHA)
        game._init_fonts()
        game.show_help = diagnostics
        game.mode = 'AUTONOMOUS' if autonomous else 'MANUAL'
        game.drawing_mode = drawing
        game.show_rays = autonomous
        if autonomous:
            game.state = VehicleState(5, 8, .7, .2, -.1, .15)
            game.current_path_idx = 45
            game.action_probs = np.array([.02]*15)
            game.action_probs[9] += .7
            game.attention_weights = np.arange(1, 65, dtype=float)
            game.attention_weights /= game.attention_weights.sum()
        game.update_hud_layout()
        game.snowfall.elapsed = 0
        game.snowfall.last_time = time.perf_counter()
        game.render()
        pygame.image.save(game.screen, str(output / f'reference-{name}.png'))
        scenes.append({'name': name, 'width': width, 'height': height, 'mode': game.mode,
                       'diagnostics': diagnostics, 'drawing': drawing, 'rays': autonomous,
                       'state': vars(game.state), 'index': game.current_path_idx,
                       'probabilities': game.action_probs.tolist(), 'attention': game.attention_weights.tolist()})
    (output / 'visual-scenes.json').write_text(json.dumps(scenes))
    pygame.quit()


def export_truck_references(output: Path):
    """Render isolated trucks so background similarity cannot hide vehicle differences."""
    from unittest.mock import patch
    from src.game.vehicle import VehicleRenderer
    from src.kinematics import VehicleState, VehicleDimensions, TractorDoubleTrailerKinematics
    pygame.init()
    poses = [(0, 0, 0), (.7, .2, -.1), (-1.1, -.55, .35), (2.2, .9, -.7),
             (3.141592653589793, 0, 0), (-2.4, -1.05, .9)]
    fixtures = []
    class Camera:
        def __init__(self, scale):
            self.scale = scale
        def project(self, x, y):
            return int(200+x*self.scale), int(180-y*self.scale)
    for scale in (6.575, 12):
        for i, (heading, beta1, beta2) in enumerate(poses):
            state = VehicleState(0, 0, heading, beta1, beta2, .12)
            surface = pygame.Surface((400, 360), pygame.SRCALPHA)
            camera = Camera(scale)
            commands = []
            original_polygon, original_lines = pygame.draw.polygon, pygame.draw.lines
            def polygon(target, color, points, width=0):
                if width:
                    for a, b in zip(points, points[1:]+points[:1]):
                        commands.append({'kind':'line','points':[a,b],'color':color,'width':width})
                else:
                    commands.append({'kind':'polygon','points':points,'color':color})
                return original_polygon(target, color, points, width)
            def lines(target, color, closed, points, width=1):
                for a, b in zip(points, points[1:]):
                    commands.append({'kind':'line','points':[a,b],'color':color,'width':width})
                return original_lines(target, color, closed, points, width)
            with patch('pygame.draw.polygon', polygon), patch('pygame.draw.lines', lines):
                VehicleRenderer().draw(surface, camera, state, VehicleDimensions(), TractorDoubleTrailerKinematics())
            name = f'truck-{scale}-{i}'
            pygame.image.save(surface, str(output / f'{name}.png'))
            fixtures.append({'name':name,'width':400,'height':360,'scale':scale,'state':vars(state),'commands':commands})
    (output / 'truck-scenes.json').write_text(json.dumps(fixtures))
    pygame.quit()
