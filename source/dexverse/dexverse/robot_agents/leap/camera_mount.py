"""Wrist camera poses for the released LEAP assets, in each palm link frame.

LEAP's palm origins sit forward and sideways from the virtual wrist. Its palm
axes also differ from Shadow's: +X points along the fingers, +Y across the
palm, and +Z out of its back. Copying Shadow's palm-local camera pose therefore
puts the housing in front of the wrist and points the lens toward +Z.

Keep the existing 8 cm rearward / 8 cm downward optical placement relative to
the wrist, but express it in LEAP coordinates. The wrist locations below are
the inverse translations of the assets' palm_wrist_joint / *_base_joint fixed
transforms, rounded to micrometre accuracy. Single and bimanual assets differ.
"""

from math import cos, sin


# ROS optical +Z looks along the fingers (+X), tilted 0.1 rad toward +Z.
# This is the existing camera rotation composed with the cyclic axis mapping
# from Shadow palm coordinates (lateral, dorsal, distal) to LEAP coordinates.
_S, _C = sin(0.05), cos(0.05)
_OPTICAL_ROT = (-(_S + _C) / 2, (_C - _S) / 2, (_S - _C) / 2, (_S + _C) / 2)


def _offset(wrist_in_palm: tuple[float, float, float]) -> dict:
    x, y, z = wrist_in_palm
    return {"pos": (x - 0.08, y, z - 0.08), "rot": _OPTICAL_ROT, "convention": "ros"}


FLOATING_LEAP_RIGHT_WRIST_CAMERA_OFFSETS = {
    "palm_lower": _offset((-0.10, -0.04, -0.01)),
}
BIMANUAL_LEAP_WRIST_CAMERA_OFFSETS = {
    "right_palm_lower": _offset((-0.098, -0.038, 0.0)),
    "left_palm_lower_left": _offset((-0.17, -0.055, 0.015)),
}
