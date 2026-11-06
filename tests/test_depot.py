import os
import unittest
from unittest.mock import patch

import numpy as np
import pygame

from src.game import JackknifeGame25D
from src.game.depot import DepotTerrain
from src.path_generator import Path


class DepotTerrainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.environment = patch.dict(os.environ, {"SDL_VIDEODRIVER": "dummy"})
        cls.environment.start()
        cls.game = JackknifeGame25D(model_path=None)

    @classmethod
    def tearDownClass(cls):
        pygame.quit()
        cls.environment.stop()

    def setUp(self):
        self.game.drawing_mode = False
        self.game.show_help = False
        self.game.load_default_path()
        self.game.update_hud_layout()

    def test_scenery_clears_route_and_starting_vehicle(self):
        g = self.game
        for path in self.route_examples():
            g.path = path
            g.generate_procedural_yard()
            g.reset_vehicle_to_path()
            terrain = g.terrain
            for i, prop in enumerate(terrain.props):
                footprint = terrain.footprint_mask(prop, margin=1)
                self.assertIsNone(terrain.reserved_mask.overlap(footprint,(0,0)))
                for other in terrain.props[:i]:
                    self.assertIsNone(terrain.footprint_mask(other,margin=1).overlap(footprint,(0,0)))

    def test_scenery_is_repeatable_for_same_path(self):
        g = self.game
        terrain = DepotTerrain()
        terrain.configure(g.path,g.kinematics,g.state)
        self.assertEqual(terrain.props,g.terrain.props)
        self.assertEqual(terrain.dock,g.terrain.dock)
        self.assertGreater(len(terrain.props),0)

    def test_base_ground_stays_stable_when_route_changes(self):
        g = self.game
        first = pygame.Surface(g.screen.get_size(),depth=32)
        second = first.copy()
        g.terrain.draw_ground(first,g.camera,g.font_mono,editing=True)
        g.path = self.route_examples()[0]
        g.generate_procedural_yard()
        g.reset_vehicle_to_path()
        g.terrain.draw_ground(second,g.camera,g.font_mono,editing=True)
        self.assertEqual(pygame.image.tobytes(first,'RGB'),pygame.image.tobytes(second,'RGB'))

    def test_crossing_road_has_no_internal_edge(self):
        g = self.game
        path = Path(np.array([-30,30,-30,30]),np.array([-20,20,20,-20]),constrain_curvature=False)
        surface = pygame.Surface(g.screen.get_size(),depth=32)
        g.terrain.draw_route(surface,g.camera,path,g.font_mono)
        center = g.camera.project(0,0)
        self.assertEqual(g.terrain.road_mask.get_at(center),1)
        self.assertEqual(g.terrain.edge_mask.get_at(center),0)

    def test_dock_follows_destination_and_heading(self):
        g = self.game
        for path in self.route_examples():
            g.path = path
            g.generate_procedural_yard()
            g.reset_vehicle_to_path()
            dock=g.terrain.dock
            assert dock is not None
            assert g.path is not None
            self.assertEqual((dock.x,dock.y),(g.path.x[-1],g.path.y[-1]))
            self.assertEqual(dock.heading,g.path.headings[-1])

    @staticmethod
    def route_examples():
        x = np.linspace(-45,45,121)
        t = np.linspace(0,2*np.pi,161)
        return [
            Path(x,np.zeros_like(x)),
            Path(x,6*np.sin(x*.07)),
            Path(30*np.sin(t),18*np.sin(2*t)),
        ]

    def test_render_reuses_baked_terrain(self):
        g = self.game
        background = g.background_surf
        with patch.object(g.terrain,'configure',side_effect=AssertionError('rebuilt scenery during frame')):
            with patch.object(g.terrain,'draw_ground',side_effect=AssertionError('rebaked terrain during frame')):
                for _ in range(3):
                    g.render()
        self.assertIs(g.background_surf,background)


if __name__ == '__main__':
    unittest.main()
