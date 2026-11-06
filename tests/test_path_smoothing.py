import unittest

import numpy as np

from src.path_generator import smooth_and_constrain_user_path


class UserPathSmoothingTests(unittest.TestCase):
    def test_valid_bends_keep_their_shape_and_endpoints(self):
        x = np.linspace(-45, 45, 121)
        y = 6*np.sin(x*np.pi/45)
        points = np.column_stack((x, y))
        path = smooth_and_constrain_user_path(points.tolist())
        np.testing.assert_allclose([path.x[0], path.y[0]], points[0], atol=1e-12)
        np.testing.assert_allclose([path.x[-1], path.y[-1]], points[-1], atol=1e-12)
        self.assertLess(np.max(np.abs(path.y-6*np.sin(path.x*np.pi/45))), .05)

    def test_corner_relaxation_preserves_distant_straights(self):
        path = smooth_and_constrain_user_path([(-50, 0), (-25, 0), (0, 0), (0, 25), (0, 50)])
        self.assertLess(np.max(np.abs(path.y[path.x < -30])), .05)
        self.assertLess(np.max(np.abs(path.x[path.y > 30])), .05)
        arc = np.insert(np.cumsum(np.hypot(np.diff(path.x), np.diff(path.y))), 0, 0)
        curvature = np.gradient(np.unwrap(path.headings), arc)
        self.assertLessEqual(np.max(np.abs(curvature)), 1/12.5+.003)

    def test_closed_loop_stays_closed_and_keeps_its_radius(self):
        theta = np.linspace(0, 2*np.pi, 161)
        points = np.column_stack((20*np.cos(theta), 20*np.sin(theta)))
        path = smooth_and_constrain_user_path(points.tolist())
        self.assertLess(np.max(np.abs(np.hypot(path.x, path.y)-20)), .05)
        np.testing.assert_allclose([path.x[0], path.y[0]], [path.x[-1], path.y[-1]], atol=1e-12)

    def test_user_lead_in_stays_straight_after_smoothing(self):
        x = np.linspace(-45,45,121)
        y = 12*np.sin((x+45)*.045)
        points = np.column_stack((x,y))
        path = smooth_and_constrain_user_path(points.tolist(),lead_in_m=12)
        prefix = np.column_stack((path.x,path.y))[path.s < 8]
        direction = prefix[-1]-prefix[0]
        offsets = prefix-prefix[0]
        cross = offsets[:,0]*direction[1]-offsets[:,1]*direction[0]
        self.assertLess(np.max(np.abs(cross)),1e-8)
        np.testing.assert_allclose([path.x[0],path.y[0]],points[0],atol=1e-12)
        np.testing.assert_allclose([path.x[-1],path.y[-1]],points[-1],atol=1e-12)

    def test_short_drawing_also_starts_straight(self):
        path = smooth_and_constrain_user_path([(0,0),(1,.3),(2,1),(3,1.2)],lead_in_m=12)
        self.assertGreaterEqual(len(path.x),4)
        self.assertLess(np.max(np.abs(path.y-.4*path.x)),1e-8)


if __name__ == "__main__":
    unittest.main()
