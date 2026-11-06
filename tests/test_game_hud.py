import os
import math
import unittest
from collections import defaultdict
from dataclasses import astuple
from unittest.mock import patch
from typing import cast

import pygame
import numpy as np

from src.game import JackknifeGame25D


class GameHUDTests(unittest.TestCase):
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
        g = self.game
        g.drawing_mode = g.is_drawing = g.is_paused = False
        g.time_multiplier = 1
        g.mode = "AUTONOMOUS"
        g.ai_save_active = False
        g.show_help = False
        g.show_rays = False
        g.load_default_path()
        self.resize(1400, 900)

    def resize(self, width, height):
        pygame.event.clear()
        pygame.event.post(pygame.event.Event(pygame.VIDEORESIZE, w=width, h=height))
        self.game.handle_input()

    def send(self, kind, **kwargs):
        pygame.event.post(pygame.event.Event(kind, **kwargs))
        self.game.handle_input()

    def test_yard_never_overlaps_chrome_when_resized(self):
        g = self.game
        for width, height in [(1400, 900), (900, 700), (640, 600), (320, 480), (1920, 1080)]:
            for diagnostics in (False, True):
                with self.subTest(width=width, diagnostics=diagnostics):
                    g.show_help = diagnostics
                    self.resize(width, height)
                    g.render()
                    for chrome in [g.hud.toolbar_rect, g.hud.status_rect, g.hud.drawer_rect]:
                        self.assertFalse(g.camera.viewport.colliderect(chrome))
                    for button in g.hud.button_rects.values():
                        self.assertTrue(g.screen.get_rect().contains(button))
                    for point in [(-80, -50), (80, 50)]:
                        self.assertTrue(g.camera.viewport.collidepoint(g.camera.project(*point)))

    def test_camera_round_trip_uses_offset_viewport(self):
        g = self.game
        g.show_help = True
        g.update_hud_layout()
        self.assertEqual(g.camera.project(0, 0), g.camera.viewport.center)
        for world in [(-50, 20), (35, -25), (0, 0)]:
            restored = g.camera.unproject_ground(*g.camera.project(*world))
            for actual, expected in zip(restored, world):
                self.assertAlmostEqual(actual, expected, delta=1/g.camera.scale)

    def test_toolbar_controls_do_not_start_drawing(self):
        g = self.game
        for speed in (2, 4, 8, 16, 1):
            self.send(pygame.MOUSEBUTTONDOWN, button=1, pos=g.hud.button_rects["speed"].center)
            self.assertEqual(g.time_multiplier, speed)
            self.assertFalse(g.drawing_mode)
        self.send(pygame.MOUSEBUTTONDOWN, button=1, pos=g.hud.button_rects["pause"].center)
        self.assertTrue(g.is_paused)
        self.assertFalse(g.drawing_mode)
        self.send(pygame.MOUSEBUTTONDOWN, button=1, pos=g.hud.button_rects["diagnostics"].center)
        self.assertTrue(g.show_help)
        self.assertFalse(g.drawing_mode)

    def test_control_mode_button_and_keyboard_stay_in_sync(self):
        g = self.game
        g.steps_until_inference = 3
        self.send(pygame.MOUSEBUTTONDOWN, button=1, pos=g.hud.button_rects["mode"].center)
        self.assertEqual(g.mode, "MANUAL")
        self.assertEqual(g.steps_until_inference, 0)
        self.assertFalse(g.drawing_mode)
        g.show_help = True
        g.update_hud_layout()
        g.render()
        self.assertGreater(g.hud.manual_steering_bar_rect.w, 0)
        self.send(pygame.KEYDOWN, key=pygame.K_m)
        self.assertEqual(g.mode, "AUTONOMOUS")
        g.render()
        self.assertGreater(g.hud.histogram_plot_rect.w, 0)

    def test_attention_rays_have_an_independent_toggle(self):
        g = self.game
        g.show_help = True
        g.update_hud_layout()
        g.render()
        self.assertGreater(g.hud.rays_button_rect.w, 0)
        self.send(pygame.MOUSEBUTTONDOWN, button=1, pos=g.hud.rays_button_rect.center)
        self.assertTrue(g.show_rays)
        self.assertFalse(g.drawing_mode)
        self.send(pygame.KEYDOWN, key=pygame.K_v)
        self.assertFalse(g.show_rays)

    def test_button_label_and_keycap_are_optically_centered(self):
        g = self.game
        rect = pygame.Rect(100,100,180,40)
        for label,key in [("AUTO","M"),("NEW ROUTE","P"),("CONTINUE","Space"),("RETRY","R")]:
            g.screen.fill((0,0,0))
            g.hud._button(rect,label,key)
            mask = pygame.mask.from_threshold(g.screen,(242,235,213),(1,1,1,255))
            mask.draw(pygame.mask.from_threshold(g.screen,(159,151,178),(1,1,1,255)),(0,0))
            mask.draw(pygame.mask.from_threshold(g.screen,(9,9,17),(1,1,1,255)),(0,0))
            components = cast(list[pygame.Rect], mask.get_bounding_rects())
            self.assertTrue(components)
            ink = components[0].unionall(components[1:])
            self.assertAlmostEqual(ink.centerx,rect.centerx,delta=1)
            self.assertAlmostEqual(ink.centery,rect.centery,delta=1)

    def test_expected_marker_triangle_matches_stem_center(self):
        g = self.game
        g.show_help = True
        g.update_hud_layout()
        g.action_probs = np.zeros(15);g.action_probs[10] = 1
        g.render()
        plot = g.hud.histogram_plot_rect
        center = g.hud.expected_steering_x
        assert center is not None
        for y in (plot.top-8,plot.top+1):
            pixels=[x for x in range(center-8,center+9) if g.screen.get_at((x,y))[:3] == (135,225,130)]
            self.assertTrue(pixels)
            self.assertEqual(min(pixels)+max(pixels),2*center)

    def test_clicks_on_status_and_diagnostics_do_not_draw(self):
        g = self.game
        g.show_help = True
        g.update_hud_layout()
        original = g.path
        for pos in [g.hud.status_rect.center, g.hud.drawer_rect.center, g.hud.brand_rect.center]:
            self.send(pygame.MOUSEBUTTONDOWN, button=1, pos=pos)
            self.assertFalse(g.drawing_mode)
            self.assertIs(g.path, original)

    def test_editor_keeps_diagnostics_and_freezes_physics(self):
        g = self.game
        g.show_help = True
        g.update_hud_layout()
        old_width = g.camera.viewport.w
        before = astuple(g.state)
        self.send(pygame.MOUSEBUTTONDOWN, button=1, pos=g.hud.button_rects["draw"].center)
        self.assertTrue(g.drawing_mode)
        self.assertGreater(g.hud.drawer_rect.w, 0)
        self.assertEqual(g.camera.viewport.w, old_width)
        g.advance_simulation(0.2, defaultdict(bool))
        self.assertEqual(astuple(g.state), before)
        self.send(pygame.KEYDOWN, key=pygame.K_ESCAPE)
        self.assertFalse(g.drawing_mode)
        self.assertEqual(g.camera.viewport.w, old_width)

    def test_histogram_axis_and_expectation_follow_bin_centers(self):
        g = self.game
        g.show_help = True
        g.update_hud_layout()
        for index, angle in [(0, -35), (7, 0), (14, 35)]:
            g.action_probs = np.zeros(15)
            g.action_probs[index] = 1
            g.render()
            ticks = dict(g.hud.axis_ticks)
            self.assertEqual(g.hud.expected_steering_x, ticks[angle])
            self.assertGreater(ticks[-35], g.hud.histogram_plot_rect.left)
            self.assertLess(ticks[35], g.hud.histogram_plot_rect.right)
            self.assertAlmostEqual(ticks[0]-ticks[-35], ticks[35]-ticks[0], delta=1)

    def test_manual_steering_bar_tracks_wheel_angle(self):
        g = self.game
        g.mode = "MANUAL"
        g.show_help = True
        g.update_hud_layout()
        for angle in (-20, 20):
            g.state.phi = math.radians(angle)
            g.render()
            bar = g.hud.manual_steering_bar_rect
            self.assertGreater(bar.w, 0)
            self.assertEqual(g.hud.histogram_plot_rect.w, 0)
            self.assertEqual(g.hud.manual_steering_cursor_x > bar.centerx, angle > 0)

    def test_off_track_detection_stops_both_driving_modes(self):
        g = self.game
        for mode in ("AUTONOMOUS", "MANUAL"):
            g.reset_vehicle_to_path()
            g.mode = mode
            assert g.path is not None
            normal = g.path.headings[0]+math.pi/2
            g.state.x0 += 12*math.cos(normal)
            g.state.y0 += 12*math.sin(normal)
            g.current_speed = -1.5
            g.update_physics()
            self.assertTrue(g.is_off_track)
            self.assertEqual(g.current_speed, 0)
            g.render()
            alert = g.hud._alert_state()
            assert alert is not None
            self.assertEqual(alert[0], "OFF TRACK")
            self.assertEqual(g.hud.alert_rect.center, g.camera.viewport.center)

    def test_terminal_messages_are_centered_and_retry_resets(self):
        g = self.game
        for condition, title in [("jackknife", "JACKKNIFED"), ("off_track", "OFF TRACK"), ("done", "DOCK SECURED")]:
            g.reset_vehicle_to_path()
            if condition == "jackknife":
                g.state.beta1 = g.dims.crit_hitch_rad+.01
            elif condition == "off_track":
                g.is_off_track = True
            else:
                g.has_arrived = True
            g.cross_track_err = 9
            g.inference_time_ms = 123
            g.render()
            alert = g.hud._alert_state()
            assert alert is not None
            self.assertEqual(alert[0], title)
            self.assertEqual(g.hud.alert_rect.center, g.camera.viewport.center)
            self.send(pygame.MOUSEBUTTONDOWN, button=1, pos=g.hud.alert_reset_rect.center)
            self.assertFalse(g.kinematics.is_jackknifed(g.state))
            self.assertFalse(g.is_off_track)
            self.assertFalse(g.has_arrived)
            self.assertEqual(g.cross_track_err, 0)
            self.assertEqual(g.inference_time_ms, 0)

    def test_footer_reset_label_has_room_and_is_clickable(self):
        g = self.game
        for width in (320, 640, 1400, 1920):
            self.resize(width, 900)
            g.render()
            available = g.hud.reset_rect.w-g.font_sm.size("R")[0]-4*g.hud.gap
            self.assertGreaterEqual(available, g.font_arcade.size("RESET")[0])
            self.assertTrue(g.hud.status_rect.contains(g.hud.reset_rect))
            g.current_path_idx = 10
            self.send(pygame.MOUSEBUTTONDOWN, button=1, pos=g.hud.reset_rect.center)
            self.assertEqual(g.current_path_idx, 0)

    def test_escape_preserves_original_path_and_vehicle(self):
        g = self.game
        old_path, old_state = g.path, astuple(g.state)
        self.send(pygame.MOUSEBUTTONDOWN, button=1, pos=g.camera.project(-20, 0))
        self.send(pygame.MOUSEMOTION, pos=g.camera.project(0, 10))
        self.send(pygame.KEYDOWN, key=pygame.K_ESCAPE)
        self.assertIs(g.path, old_path)
        self.assertEqual(astuple(g.state), old_state)
        self.assertFalse(g.is_drawing)

    def test_release_commits_path_using_yard_coordinates(self):
        g = self.game
        old_path = g.path
        g.start_drawing_mode()
        points = [(-30, 0), (-15, 2), (0, 3), (15, 2), (30, 0)]
        self.send(pygame.MOUSEBUTTONDOWN, button=1, pos=g.camera.project(*points[0]))
        for point in points[1:]:
            self.send(pygame.MOUSEMOTION, pos=g.camera.project(*point))
        for expected, actual in zip(points, g.drawing_points):
            for a, b in zip(expected, actual):
                self.assertAlmostEqual(a, b, delta=1/g.camera.scale)
        self.send(pygame.MOUSEBUTTONUP, button=1, pos=g.camera.project(*points[-1]))
        self.assertFalse(g.drawing_mode)
        self.assertIsNot(g.path, old_path)
        self.assertEqual(g.simulation_accumulator, 0)


if __name__ == "__main__":
    unittest.main()
