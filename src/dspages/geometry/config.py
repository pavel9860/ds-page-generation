"""Tunable numbers of this package, one section per module."""

# strip
MAX_STEPS = 200
MAX_DPHI = 0.2
STEP_TOL = 1e-5  # m: stop when no node moves more than this in a step (a contact switching on and off
REG = 1e-9

# surface
ISO_PASSES = 2  # row and column passes that give a creased sheet its lengths back (strain < 1e-4)
