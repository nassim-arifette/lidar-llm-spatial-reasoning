"""Spatial predicates, with the convention written down once and only once.

Every directional word used by the question parser ("front", "left", "near",
...) resolves to exactly one function here. A geometry error is therefore
localisable: if "front" disagrees with the dataset's notion of front, exactly
one constant in this file is wrong.

Conventions
-----------
* Ego frame: x forward, y left, z up (see ``objects.py``).
* Directions are defined by **bearing sectors**, not by axis-aligned
  half-planes. A car 40 m ahead and 2 m to the left is "front", not
  "front left"; a half-plane test would label it both. The sector width is an
  explicit parameter.
* ``near`` / ``far`` use horizontal range, not 3D distance: the height of a
  bounding-box centre is dominated by the object class (a bus centre sits
  higher than a traffic cone) and carries no useful information about
  proximity on the road plane.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from .objects import Object3D

# --------------------------------------------------------------------------- #
# Tunable, explicit geometry constants
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class RegionConfig:
    """All geometry thresholds in one place, so they can be reported and swept.

    Attributes
    ----------
    sector_half_width:
        Half-angle (radians) of the "front" / "back" / "left" / "right"
        sectors. The default of pi/8 gives 45-degree sectors separated by
        45-degree diagonal sectors, so the eight directions tile the plane.
    near_radius:
        Horizontal distance (m) below which an object counts as "near".
    far_radius:
        Horizontal distance (m) above which an object counts as "far".
    min_range:
        Objects closer than this to the ego origin have an unstable bearing
        (a 10 cm error flips the sign of y), so they are excluded from
        directional queries and reported separately.
    """

    sector_half_width: float = math.pi / 8.0
    near_radius: float = 20.0
    far_radius: float = 35.0
    min_range: float = 1.0


DEFAULT_REGION_CONFIG = RegionConfig()


# --------------------------------------------------------------------------- #
# Bearing sectors
# --------------------------------------------------------------------------- #

#: Centre bearing of each named direction, in the ego frame.
#: 0 is straight ahead, positive towards the left.
_DIRECTION_BEARING: Dict[str, float] = {
    "front": 0.0,
    "front_left": math.pi / 4.0,
    "left": math.pi / 2.0,
    "back_left": 3.0 * math.pi / 4.0,
    "back": math.pi,
    "back_right": -3.0 * math.pi / 4.0,
    "right": -math.pi / 2.0,
    "front_right": -math.pi / 4.0,
}

DIRECTIONS: tuple = tuple(_DIRECTION_BEARING.keys())

#: Words a question may use for each direction.
DIRECTION_ALIASES: Dict[str, str] = {
    "front": "front", "ahead": "front", "forward": "front", "in_front": "front",
    "back": "back", "behind": "back", "rear": "back",
    "left": "left",
    "right": "right",
    "front_left": "front_left", "left_front": "front_left",
    "front_right": "front_right", "right_front": "front_right",
    "back_left": "back_left", "left_back": "back_left", "rear_left": "back_left",
    "back_right": "back_right", "right_back": "back_right", "rear_right": "back_right",
}


def _angle_difference(a: float, b: float) -> float:
    """Smallest absolute difference between two angles, in ``[0, pi]``."""
    d = (a - b + math.pi) % (2.0 * math.pi) - math.pi
    return abs(d)


def in_direction(
    obj: Object3D,
    direction: str,
    config: RegionConfig = DEFAULT_REGION_CONFIG,
    origin: Optional[Sequence[float]] = None,
    reference_yaw: float = 0.0,
) -> bool:
    """Is ``obj`` inside the named sector?

    ``origin`` and ``reference_yaw`` let the same sectors be used relative to
    another object instead of the ego vehicle (see :func:`relative_direction`).
    """
    name = DIRECTION_ALIASES.get(direction, direction)
    if name not in _DIRECTION_BEARING:
        raise KeyError(f"unknown direction {direction!r}; known: {sorted(DIRECTIONS)}")

    dx = obj.center[0] - (origin[0] if origin is not None else 0.0)
    dy = obj.center[1] - (origin[1] if origin is not None else 0.0)
    if math.hypot(dx, dy) < config.min_range:
        return False

    bearing = math.atan2(dy, dx) - reference_yaw
    return _angle_difference(bearing, _DIRECTION_BEARING[name]) <= config.sector_half_width


def relative_direction(
    obj: Object3D,
    reference: Object3D,
    direction: str,
    config: RegionConfig = DEFAULT_REGION_CONFIG,
    frame: str = "ego",
) -> bool:
    """Direction of ``obj`` relative to another object.

    ``frame`` makes the ambiguity of "left of the truck" explicit:

    * ``"ego"``   -- left as seen from the ego vehicle (the viewer's left).
    * ``"object"`` -- the truck's own left, i.e. rotated by the truck's yaw.

    nuScenes-QA uses the viewer-centric reading, so ``"ego"`` is the default,
    but the alternative is implemented because a system that silently picks one
    convention cannot be audited.
    """
    if frame not in ("ego", "object"):
        raise ValueError("frame must be 'ego' or 'object'")
    ref_yaw = reference.yaw if frame == "object" else 0.0
    return in_direction(obj, direction, config,
                        origin=reference.center, reference_yaw=ref_yaw)


# --------------------------------------------------------------------------- #
# Range predicates
# --------------------------------------------------------------------------- #

def is_near(obj: Object3D, config: RegionConfig = DEFAULT_REGION_CONFIG) -> bool:
    return obj.range_xy <= config.near_radius


def is_far(obj: Object3D, config: RegionConfig = DEFAULT_REGION_CONFIG) -> bool:
    return obj.range_xy >= config.far_radius


def within_radius(obj: Object3D, radius: float) -> bool:
    return obj.range_xy <= radius


# --------------------------------------------------------------------------- #
# Region objects usable by the program executor
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Region:
    """A named, serialisable spatial constraint.

    ``kind`` is one of ``"any"``, ``"direction"``, ``"near"``, ``"far"``,
    ``"radius"``, ``"relative"``.
    """

    kind: str = "any"
    direction: Optional[str] = None
    radius: Optional[float] = None
    reference_category: Optional[str] = None
    frame: str = "ego"

    def describe(self) -> str:
        if self.kind == "any":
            return "anywhere in the scene"
        if self.kind == "direction":
            return f"{self.direction.replace('_', ' ')} of the ego vehicle"
        if self.kind == "near":
            return "near the ego vehicle"
        if self.kind == "far":
            return "far from the ego vehicle"
        if self.kind == "radius":
            return f"within {self.radius:g} m of the ego vehicle"
        if self.kind == "relative":
            return (f"{self.direction.replace('_', ' ')} of the "
                    f"{self.reference_category} ({self.frame} frame)")
        return self.kind

    def predicate(
        self,
        config: RegionConfig = DEFAULT_REGION_CONFIG,
        reference: Optional[Object3D] = None,
    ) -> Callable[[Object3D], bool]:
        """Compile the region into a single-object test."""
        if self.kind == "any":
            return lambda o: True
        if self.kind == "direction":
            return lambda o: in_direction(o, self.direction, config)
        if self.kind == "near":
            return lambda o: is_near(o, config)
        if self.kind == "far":
            return lambda o: is_far(o, config)
        if self.kind == "radius":
            r = float(self.radius if self.radius is not None else config.near_radius)
            return lambda o: within_radius(o, r)
        if self.kind == "relative":
            if reference is None:
                raise ValueError("relative region requires a resolved reference object")
            return lambda o: (o is not reference) and relative_direction(
                o, reference, self.direction, config, self.frame
            )
        raise KeyError(f"unknown region kind {self.kind!r}")


ANY_REGION = Region(kind="any")


def select(
    objects: Sequence[Object3D],
    region: Region,
    config: RegionConfig = DEFAULT_REGION_CONFIG,
    reference: Optional[Object3D] = None,
) -> List[Object3D]:
    """Apply a region to a list of objects."""
    test = region.predicate(config, reference)
    return [o for o in objects if test(o)]


def direction_of(
    obj: Object3D,
    config: RegionConfig = DEFAULT_REGION_CONFIG,
) -> Optional[str]:
    """Name the sector an object falls in, or ``None`` if it is too close.

    Used by the counterfactual generator to check that a moved object really
    landed where it was meant to.
    """
    if obj.range_xy < config.min_range:
        return None
    bearing = obj.bearing
    best, best_delta = None, math.inf
    for name, centre in _DIRECTION_BEARING.items():
        delta = _angle_difference(bearing, centre)
        if delta < best_delta:
            best, best_delta = name, delta
    return best if best_delta <= config.sector_half_width else None


def sample_position_in_direction(
    direction: str,
    distance: float,
    config: RegionConfig = DEFAULT_REGION_CONFIG,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """A point at ``distance`` metres, inside the named sector.

    The bearing is drawn strictly inside the sector (90% of its half-width) so
    that a counterfactual move does not land on the boundary, where the answer
    would depend on floating-point noise.
    """
    rng = rng or np.random.default_rng()
    name = DIRECTION_ALIASES.get(direction, direction)
    centre = _DIRECTION_BEARING[name]
    offset = rng.uniform(-0.9, 0.9) * config.sector_half_width
    bearing = centre + offset
    return np.array(
        [distance * math.cos(bearing), distance * math.sin(bearing), 0.0],
        dtype=np.float32,
    )
