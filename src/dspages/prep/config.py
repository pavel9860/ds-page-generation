"""Tunable numbers of this package, one section per module."""

# manifest
FILE_DONE = 1 << 30  # pages counted as taken from such a file

# realmaps
SHEET = (210.0, 297.0)  # mm (width, height)
MESH = (256, 360)  # nodes (across, down)
ROUNDS = 4  # column and row passes of `even`
DUP_MM = 1.0  # rms height difference [mm] under which two shapes are one group
SPLIT = (("val", 0.05), ("test", 0.05))
