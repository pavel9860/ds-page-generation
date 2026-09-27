import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from elastica.two_support import solve_two_support


class TestTwoSupportSweep(unittest.TestCase):
    def test_on_or_above_table(self):
        for length in (0.210, 0.297):
            for p1 in (0.05, 0.15, 0.30):
                for p2 in (0.70, 0.85, 0.95):
                    for h1 in (0.005, 0.02, 0.04):
                        for h2 in (0.005, 0.02, 0.04):
                            with self.subTest(length=length, p1=p1, p2=p2, h1=h1, h2=h2):
                                x, z = solve_two_support(p1 * length, h1, p2 * length, h2, length)
                                self.assertGreaterEqual(z.min(), -3e-6)
                                self.assertAlmostEqual(np.sum(np.hypot(np.diff(x), np.diff(z))), length, delta=2e-4)


if __name__ == "__main__":
    unittest.main()
