import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from analytic_tip_flat import solve_arm_analytic
from elastica_continuation import D, _rk4_fixed_3

Q = 0.7848
DEGS = range(1, 82, 10)
ARMS_MM = range(10, 401, 10)
Z_PEN = 3e-6
Z_END = 1e-5
V_END = 1e-2
THETA_END = 1e-3


def trajectory(arm, n=400):
    return _rk4_fixed_3(arm["theta0"], arm["v0"], arm["ell"], Q, D, n_steps=n,
                        r_tip=arm.get("r_tip", 0.0))


class TestArmSweep(unittest.TestCase):
    def test_all_arms(self):
        for deg in DEGS:
            for ell_mm in ARMS_MM:
                with self.subTest(deg=deg, ell_mm=ell_mm):
                    arm = solve_arm_analytic(math.radians(deg), ell_mm / 1000.0)
                    tr = trajectory(arm)
                    theta1, v1, z1 = tr[-1]
                    self.assertGreaterEqual(tr[:, 2].min(), -Z_PEN)
                    self.assertLess(abs(v1), V_END)
                    self.assertLessEqual(arm["ell"], ell_mm / 1000.0 + 1e-12)
                    if arm["branch"] in ("tip", "flat"):
                        self.assertLess(abs(z1), Z_END)
                    if arm["branch"] == "flat":
                        self.assertLess(abs(theta1), THETA_END)
                        self.assertGreater(arm["ell"], 0.0)


class TestRaisedClampSweep(unittest.TestCase):
    def test_all_arms_above_plane(self):
        for h_mm in (5, 20, 40):
            plane = -h_mm / 1000.0
            for deg in range(0, 86, 5):
                for ell_mm in range(10, 301, 10):
                    with self.subTest(h_mm=h_mm, deg=deg, ell_mm=ell_mm):
                        arm = solve_arm_analytic(math.radians(deg), ell_mm / 1000.0, plane=plane)
                        tr = trajectory(arm)
                        theta1, v1, z1 = tr[-1]
                        self.assertGreaterEqual(tr[:, 2].min(), plane - Z_PEN)
                        self.assertLess(abs(v1), V_END)
                        self.assertLessEqual(arm["ell"], ell_mm / 1000.0 + 1e-12)
                        if arm["branch"] in ("tip", "flat"):
                            self.assertLess(abs(z1 - plane), Z_END)
                        if arm["branch"] == "flat":
                            self.assertLess(abs(theta1), THETA_END)


class TestArmRegressions(unittest.TestCase):
    def test_flat_touches_down(self):
        for deg, ell_mm in ((31, 230), (7, 200), (20, 297), (41, 300)):
            with self.subTest(deg=deg, ell_mm=ell_mm):
                arm = solve_arm_analytic(math.radians(deg), ell_mm / 1000.0)
                self.assertEqual(arm["branch"], "flat")
                theta1, v1, z1 = trajectory(arm)[-1]
                self.assertLess(abs(z1), Z_END)
                self.assertLess(abs(theta1), THETA_END)

    def test_81deg_300mm_rests_on_table(self):
        arm = solve_arm_analytic(math.radians(81), 0.300)
        self.assertIn(arm["branch"], ("tip", "flat"))
        tr = trajectory(arm)
        self.assertLess(abs(tr[-1, 2]), Z_END)
        self.assertGreaterEqual(tr[:, 2].min(), -Z_PEN)


if __name__ == "__main__":
    unittest.main()
