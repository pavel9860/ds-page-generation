"""Tunable numbers of this package, one section per module."""

# generate
BG_STEP = 4
SHADOW_STEP = 2  # shadow maps from every 2nd node: the 384 px maps cannot resolve the full grid
MIP_LEVELS = (0.0, 0.5, 1.0, 2.0)  # px, page blurs blended by how much a photo pixel shrinks the page
FLAT_ROUNDS = 6  # _flat_part rounds at most
MIP_STEP = 4  # the shrink is measured on every 4th pixel (it changes slowly)

# photo
AREA_SAMPLES = 6
SHADOW_PX = 384
SHADOW_BIAS_MM = 0.6
PLAIN_PAPER_PROB = 0.05
PLAIN_GRAY_PROB = 0.05
TONE_LEVELS = 4096
