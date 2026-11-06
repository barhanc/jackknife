import math
import unittest

import pygame

from src.game import TopDownCamera
from src.game.lighting import TruckLighting
from src.kinematics import TractorDoubleTrailerKinematics, VehicleDimensions, VehicleState


class TruckLightingTests(unittest.TestCase):
    def setUp(self):
        self.lights = TruckLighting()
        self.camera = TopDownCamera(600,600)
        self.camera.scale = 12
        self.dims = VehicleDimensions()
        self.kinematics = TractorDoubleTrailerKinematics(self.dims)

    def render(self, state, reversing=False):
        layer = pygame.Surface((600,600),pygame.SRCALPHA,32)
        self.lights.draw_ground(layer,self.camera,state,self.dims,self.kinematics.get_trailer2_axle(state),reversing)
        return layer

    def test_headlight_field_follows_heading_and_fades(self):
        for heading in (0,math.pi/2):
            state = VehicleState(theta0=heading)
            layer = self.render(state)
            x,y = self.lights.headlights(state,self.dims)[0]
            def alpha(distance):
                pixel=self.camera.project(x+distance*math.cos(heading),y+distance*math.sin(heading))
                return layer.get_at(pixel).a
            self.assertGreater(alpha(2),alpha(9))
            self.assertGreater(alpha(9),alpha(-3))
            self.assertEqual(alpha(15),0)

    def test_reverse_field_only_appears_when_reversing(self):
        state = VehicleState()
        parked = self.render(state)
        reversing = self.render(state,reversing=True)
        x,y,heading=self.kinematics.get_trailer2_axle(state)
        behind=self.camera.project(x-5*math.cos(heading),y-5*math.sin(heading))
        self.assertGreater(reversing.get_at(behind).a,parked.get_at(behind).a)

    def test_light_sprites_are_reused_until_zoom_changes(self):
        self.render(VehicleState())
        sprite=self.lights._sprites['head'][0]
        self.render(VehicleState(theta0=.4))
        self.assertIs(self.lights._sprites['head'][0],sprite)
        self.camera.scale *= 1.5
        self.render(VehicleState())
        self.assertIsNot(self.lights._sprites['head'][0],sprite)


if __name__ == '__main__':
    unittest.main()
