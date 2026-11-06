import math
import unittest

import numpy as np

from src.game.vehicle import VehicleRenderer
from src.kinematics import VehicleDimensions


class VehicleClearanceTests(unittest.TestCase):
    def test_lead_cargo_body_clears_sleeper_through_articulation_range(self):
        renderer = VehicleRenderer()
        dims = VehicleDimensions()
        cab = np.array(renderer.body_corners(1.95,dims.cab_width))+[1.65,0]
        lead = np.array(renderer.body_corners(renderer.LEAD_LENGTH,dims.trailer1_width,renderer.LEAD_NOSE_CHAMFER))+[renderer.LEAD_CENTER,0]
        for angle in np.linspace(-dims.crit_hitch_rad,dims.crit_hitch_rad,141):
            c,s=math.cos(angle),math.sin(angle)
            rotated=lead@np.array([[c,s],[-s,c]])
            separated=False
            for polygon in (cab,rotated):
                for dx,dy in np.roll(polygon,-1,axis=0)-polygon:
                    axis=np.array([-dy,dx])
                    a,b=cab@axis,rotated@axis
                    if a.max()<b.min() or b.max()<a.min():
                        separated=True
                        break
                if separated:
                    break
            with self.subTest(degrees=math.degrees(angle)):
                self.assertTrue(separated,'Lead cargo body intersects the sleeper cab')
