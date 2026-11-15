"""Tunable numbers of this package, one section per module."""

# cockle
PAD = 64  # nodes: ~ the coarsest sigma
MC = 16  # noise realisations per gain entry

# deep
FIELD_MM = 2.0

# shallow
KINDS = {"single": 0.45, "chain": 0.15, "T": 0.15, "star": 0.10, "V": 0.10, "X": 0.05}
CH = 8  # segments per culling chunk
