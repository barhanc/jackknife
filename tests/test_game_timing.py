import math
import unittest
from collections import defaultdict
from dataclasses import astuple

import pygame

from src.game import JackknifeGame25D
from src.kinematics import TractorDoubleTrailerKinematics, VehicleDimensions, VehicleState


class TimingGame(JackknifeGame25D):
    policy_calls: int


class GameTimingTests(unittest.TestCase):
    def make_game(self, multiplier=1, mode="AUTONOMOUS"):
        # Exercise controls, scheduling and real kinematics without a window/model.
        game = TimingGame.__new__(TimingGame)
        game.dims = VehicleDimensions()
        game.kinematics = TractorDoubleTrailerKinematics(game.dims)
        game.state = VehicleState()
        game.dt = 0.02
        game.simulation_accumulator = 0.0
        game.steps_until_inference = 0
        game.time_multiplier = multiplier
        game.mode = mode
        game.ai_save_active = False
        game.is_paused = False
        game.has_arrived = False
        game.is_off_track = False
        game.path = None
        game.current_speed = 0.0
        game.target_steer = 0.0
        game.policy_calls = 0

        def policy():
            game.policy_calls += 1
            game.target_steer = 0.1 * math.sin(game.policy_calls * game.dt)

        game.step_autonomous_policy = policy
        return game

    def advance(self, game, seconds, frames, keys=None):
        if keys is None:
            keys = defaultdict(bool)
        for _ in range(frames):
            game.advance_simulation(seconds / frames, keys)

    def test_frame_rate_does_not_change_simulation(self):
        reference = self.make_game()
        self.advance(reference, 1.0, 50)
        for fps in (15, 30, 60, 144):
            with self.subTest(fps=fps):
                game = self.make_game()
                self.advance(game, 1.0, fps)
                self.assertEqual(game.policy_calls, 13)
                self.assertEqual(astuple(game.state), astuple(reference.state))

    def test_fast_forward_preserves_policy_and_trajectory(self):
        reference = self.make_game()
        self.advance(reference, 2.0, 120)
        for multiplier in (2, 4, 8, 16):
            with self.subTest(multiplier=multiplier):
                game = self.make_game(multiplier)
                self.advance(game, 2.0 / multiplier, 15)
                self.assertEqual(game.policy_calls, 25)
                self.assertEqual(astuple(game.state), astuple(reference.state))

    def test_uneven_frame_times_retain_fractional_steps(self):
        reference = self.make_game()
        self.advance(reference, 1.0, 50)
        game = self.make_game()
        for _ in range(10):
            for elapsed in (0.003, 0.027, 0.011, 0.059):
                game.advance_simulation(elapsed, defaultdict(bool))
        self.assertEqual(game.policy_calls, 13)
        self.assertEqual(astuple(game.state), astuple(reference.state))

    def test_manual_steering_and_coasting_are_frame_independent(self):
        keys = defaultdict(bool, {pygame.K_w: True, pygame.K_a: True})
        reference = self.make_game(mode="MANUAL")
        self.advance(reference, 1.0, 50, keys)
        self.advance(reference, 1.0, 50)
        for multiplier, frames in ((1, 30), (1, 144), (2, 15), (4, 15), (8, 15), (16, 15)):
            with self.subTest(multiplier=multiplier, frames=frames):
                game = self.make_game(multiplier, "MANUAL")
                self.advance(game, 1.0 / multiplier, frames, keys)
                self.advance(game, 1.0 / multiplier, frames)
                self.assertEqual(astuple(game.state), astuple(reference.state))
                self.assertEqual(game.target_steer, reference.target_steer)
                self.assertEqual(game.current_speed, reference.current_speed)

    def test_pause_discards_elapsed_time_and_pending_steps(self):
        game = self.make_game()
        game.advance_simulation(0.01, defaultdict(bool))
        game.is_paused = True
        game.advance_simulation(10.0, defaultdict(bool))
        self.assertEqual(game.simulation_accumulator, 0.0)
        self.assertEqual(game.policy_calls, 0)
        game.is_paused = False
        game.advance_simulation(0.02, defaultdict(bool))
        self.assertEqual(game.policy_calls, 1)

    def test_speed_change_retains_partial_simulated_step(self):
        game = self.make_game()
        game.advance_simulation(0.01, defaultdict(bool))
        game.time_multiplier = 8
        game.advance_simulation(0.01 / 8, defaultdict(bool))
        self.assertEqual(game.policy_calls, 1)
        self.assertAlmostEqual(game.simulation_accumulator, 0.0)

    def test_long_stall_has_bounded_catch_up(self):
        game = self.make_game()
        game.advance_simulation(10.0, defaultdict(bool))
        self.assertEqual(game.policy_calls, 3)
        self.assertAlmostEqual(game.simulation_accumulator, 0.01)


if __name__ == "__main__":
    unittest.main()
