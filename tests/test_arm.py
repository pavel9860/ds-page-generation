import math
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from elastica.arm import solve_arm
from elastica.ode import D, rk4_3

Q = 0.7848


def profile_trajectory(theta0, arm, n=2000):
    """Curved-part (theta, v, z) trajectory for a solved arm dict."""
    r_tip = arm.get("r_tip", 0.0)
    return rk4_3(theta0, arm["v0"], arm["ell"], Q, D, n_steps=n, r_tip=r_tip)


class TestNoLift(unittest.TestCase):
    """The curved part must never sit below the support plane -- and, for
    the flat branch specifically, must not end up detached above it either
    (the bug this suite was written to catch: a stale contact-force lookup
    silently used 0.0 instead of the real touchdown force, leaving the
    solved curve floating off the plane at its far end)."""

    def test_no_penetration_below_plane(self):
        for deg in (20, 41, 61, 81):
            theta0 = math.radians(deg)
            for ell_mm in range(50, 301, 25):
                arm = solve_arm(theta0, ell_mm / 1000.0, plane=0.0)
                traj = profile_trajectory(theta0, arm)
                zmin = traj[:, 2].min()
                self.assertGreaterEqual(
                    zmin, -2e-6,
                    f"deg={deg} ell={ell_mm}mm branch={arm['branch']}: "
                    f"curve dips {zmin*1e3:.4f}mm below the plane")

    def test_flat_branch_touches_down_not_lifted(self):
        # The case that motivated this suite: theta0=61deg, ell=300mm.
        theta0 = math.radians(61.0)
        arm = solve_arm(theta0, 0.300, plane=0.0)
        self.assertEqual(arm["branch"], "flat")
        traj = profile_trajectory(theta0, arm)
        z_end = traj[-1, 2]
        self.assertLess(
            abs(z_end), 1e-6,
            f"flat branch's curved part should end AT the plane, got z={z_end*1e3:.4f}mm "
            "-- a lifted end means the contact force wasn't applied")

    def test_r_tip_key_present_for_every_contacting_branch(self):
        # Both the tip and flat branches place a concentrated contact
        # force somewhere along the curve; whichever branch it is, callers
        # that integrate the curved part need it under the same key.
        theta0 = math.radians(61.0)
        for ell_mm, expected_branch in ((280, "tip"), (300, "flat")):
            arm = solve_arm(theta0, ell_mm / 1000.0, plane=0.0)
            self.assertEqual(arm["branch"], expected_branch)
            self.assertIn("r_tip", arm,
                          f"branch={arm['branch']} is missing the r_tip key")


class TestSupportPosition(unittest.TestCase):
    """Touchdown location (and the state at it) must vary continuously as
    arm length crosses the tip -> flat switch -- the discontinuous jump
    this suite guards against was the original symptom of the bug."""

    def _touchdown_mm(self, theta0, arm):
        traj = profile_trajectory(theta0, arm, n=4000)
        return arm["ell"] * 1e3, traj[-1]

    def test_touchdown_continuous_across_tip_flat_switch(self):
        theta0 = math.radians(61.0)
        lens_mm = list(range(260, 301, 5))
        touchdowns = []
        for ell_mm in lens_mm:
            arm = solve_arm(theta0, ell_mm / 1000.0, plane=0.0)
            td_mm, _ = self._touchdown_mm(theta0, arm)
            touchdowns.append(td_mm)

        # touchdown must never move backward by more than a small amount
        # as arm length grows a little -- a multi-mm jump backward is
        # exactly the discontinuity this suite was written to catch.
        for i in range(1, len(touchdowns)):
            step = touchdowns[i] - touchdowns[i - 1]
            self.assertGreater(
                step, -1.0,
                f"touchdown jumped from {touchdowns[i-1]:.2f}mm to {touchdowns[i]:.2f}mm "
                f"going from {lens_mm[i-1]}mm to {lens_mm[i]}mm arm length")

    def test_moment_vanishes_at_touchdown(self):
        # Both branches' free boundary (true tip, or the flat branch's
        # touchdown point) must be moment-free (v ~ 0): nothing supplies a
        # concentrated external moment there.
        theta0 = math.radians(61.0)
        for ell_mm in (250, 280, 290, 300):
            arm = solve_arm(theta0, ell_mm / 1000.0, plane=0.0)
            traj = profile_trajectory(theta0, arm)
            v_end = traj[-1, 1]
            self.assertLess(
                abs(v_end), 1e-3,
                f"ell={ell_mm}mm branch={arm['branch']}: moment at the free "
                f"boundary is {v_end:.4f}, expected ~0")

    def test_touchdown_matches_arm_length_before_flat_onset(self):
        # Below the tip->flat switch, the whole arm is curved: touchdown
        # (the tip itself) must equal the full arm length.
        theta0 = math.radians(61.0)
        for ell_mm in (200, 230, 260, 280):
            arm = solve_arm(theta0, ell_mm / 1000.0, plane=0.0)
            self.assertEqual(arm["branch"], "tip")
            self.assertAlmostEqual(arm["ell"] * 1e3, ell_mm, places=6)


class TestArcLengthConservation(unittest.TestCase):
    """The beam is inextensible: curved + flat length must equal the
    material length fed in, for every branch."""

    def test_total_length_conserved(self):
        for deg in (10, 41, 61, 81):
            theta0 = math.radians(deg)
            for ell_mm in (60, 150, 220, 300):
                ell = ell_mm / 1000.0
                arm = solve_arm(theta0, ell, plane=0.0)
                curved = arm["ell"]
                flat_len = max(ell - curved, 0.0) if arm["branch"] == "flat" else 0.0
                self.assertAlmostEqual(curved + flat_len, ell, places=9)


if __name__ == "__main__":
    unittest.main()
