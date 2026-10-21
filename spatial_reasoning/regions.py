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
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .objects import Object3D

# --------------------------------------------------------------------------- #
# Tunable, explicit geometry constants
# --------------------------------------------------------------------------- #

#: Sector layouts. A direction maps to a half-open bearing interval
#: ``(low, high]`` in degrees, measured from the ego vehicle's forward
#: direction, counter-clockwise positive (so positive is to the left).
#:
#: ``octant`` is the layout this package used first: eight 45-degree sectors
#: tiling the plane. It is self-consistent and fine for synthetic data.
#:
#: ``nuscenes_qa`` is the benchmark's own definition (arXiv:2305.14836, eq. 2):
#: **six** 60-degree relations, with no standalone "left" or "right". Using the
#: octant layout to answer nuScenes-QA puts an object at 25 degrees in
#: "front left" where the benchmark says "front", so a perfectly correct
#: program still selects the wrong objects. Any run against nuScenes-QA must
#: use this layout.
SECTOR_LAYOUTS: Dict[str, Dict[str, Tuple[float, float]]] = {
    "octant": {
        "front":       (-22.5, 22.5),
        "front_left":  (22.5, 67.5),
        "left":        (67.5, 112.5),
        "back_left":   (112.5, 157.5),
        "back":        (157.5, -157.5),   # wraps through 180
        "back_right":  (-157.5, -112.5),
        "right":       (-112.5, -67.5),
        "front_right": (-67.5, -22.5),
    },
    "nuscenes_qa": {
        "front":       (-30.0, 30.0),
        "front_left":  (30.0, 90.0),
        "back_left":   (90.0, 150.0),
        "back":        (150.0, -150.0),   # wraps through 180
        "back_right":  (-150.0, -90.0),
        "front_right": (-90.0, -30.0),
    },
}

DEFAULT_LAYOUT = "octant"


@dataclass(frozen=True)
class RegionConfig:
    """All geometry thresholds in one place, so they can be reported and swept.

    Attributes
    ----------
    layout:
        Which entry of :data:`SECTOR_LAYOUTS` defines the directions. This is
        the single most consequential setting in the package: it decides which
        objects a directional query selects.
    near_radius:
        Horizontal distance (m) below which an object counts as "near".
    far_radius:
        Horizontal distance (m) above which an object counts as "far".
    min_range:
        Objects closer than this to the ego origin have an unstable bearing
        (a 10 cm error flips the sign of y), so they are excluded from
        directional queries and reported separately.
    """

    layout: str = DEFAULT_LAYOUT
    near_radius: float = 20.0
    far_radius: float = 35.0
    min_range: float = 1.0

    @property
    def sectors(self) -> Dict[str, Tuple[float, float]]:
        try:
            return SECTOR_LAYOUTS[self.layout]
        except KeyError as exc:
            raise KeyError(
                f"unknown sector layout {self.layout!r}; "
                f"known: {sorted(SECTOR_LAYOUTS)}"
            ) from exc

    def directions(self) -> Tuple[str, ...]:
        return tuple(self.sectors)


DEFAULT_REGION_CONFIG = RegionConfig()

#: Config to use for anything evaluated against nuScenes-QA.
NUSCENES_QA_REGION_CONFIG = RegionConfig(layout="nuscenes_qa")


# --------------------------------------------------------------------------- #
# Bearing sectors
# --------------------------------------------------------------------------- #

#: Every direction name any layout uses. Individual layouts expose a subset;
#: ``RegionConfig.directions()`` is what callers should enumerate.
DIRECTIONS: tuple = tuple(SECTOR_LAYOUTS["octant"].keys())

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


def _normalise_degrees(angle: float) -> float:
    """Wrap an angle in degrees into ``(-180, 180]``."""
    wrapped = (angle + 180.0) % 360.0 - 180.0
    return 180.0 if wrapped == -180.0 else wrapped


def _in_interval(theta: float, low: float, high: float) -> bool:
    """Is ``theta`` in the half-open interval ``(low, high]``, degrees?

    The interval may wrap through 180 (the "back" sector in every layout), so
    the two cases are handled separately rather than with a single comparison.
    """
    if low < high:
        return low < theta <= high
    return theta > low or theta <= high


def in_direction(
    obj: Object3D,
    direction: str,
    config: RegionConfig = DEFAULT_REGION_CONFIG,
    origin: Optional[Sequence[float]] = None,
    reference_yaw: float = 0.0,
) -> bool:
    """Is ``obj`` inside the named sector of ``config``'s layout?

    ``origin`` and ``reference_yaw`` let the same sectors be used relative to
    another object instead of the ego vehicle (see :func:`relative_direction`).
    """
    name = DIRECTION_ALIASES.get(direction, direction)
    sectors = config.sectors
    if name not in sectors:
        raise KeyError(
            f"direction {direction!r} does not exist in layout "
            f"{config.layout!r}; available: {sorted(sectors)}"
        )

    dx = obj.center[0] - (origin[0] if origin is not None else 0.0)
    dy = obj.center[1] - (origin[1] if origin is not None else 0.0)
    if math.hypot(dx, dy) < config.min_range:
        return False

    theta = _normalise_degrees(math.degrees(math.atan2(dy, dx) - reference_yaw))
    low, high = sectors[name]
    return _in_interval(theta, low, high)


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

    The layouts tile the plane, so at most one name matches; the loop returns
    the first and never falls back to a "nearest" sector, which would quietly
    place a boundary object inside a sector it does not belong to.
    """
    if obj.range_xy < config.min_range:
        return None
    theta = _normalise_degrees(math.degrees(obj.bearing))
    for name, (low, high) in config.sectors.items():
        if _in_interval(theta, low, high):
            return name
    return None


def sector_span(name: str, config: RegionConfig = DEFAULT_REGION_CONFIG) -> float:
    """Angular width of a sector, in degrees."""
    low, high = config.sectors[name]
    return (high - low) % 360.0 or 360.0


def sample_position_in_direction(
    direction: str,
    distance: float,
    config: RegionConfig = DEFAULT_REGION_CONFIG,
    rng: Optional[np.random.Generator] = None,
    margin_fraction: float = 0.05,
) -> np.ndarray:
    """A point at ``distance`` metres, strictly inside the named sector.

    The bearing is kept away from both edges by ``margin_fraction`` of the
    sector width, so a counterfactual move never lands on a boundary where the
    reference answer would depend on floating-point rounding.

    The centre is derived from the interval rather than assumed: under the
    nuScenes-QA layout ``front_left`` spans (30, 90] and is centred on 60
    degrees, not on 45. Hard-coding 45 was exactly the bug this fixes.
    """
    rng = rng or np.random.default_rng()
    name = DIRECTION_ALIASES.get(direction, direction)
    if name not in config.sectors:
        raise KeyError(
            f"direction {direction!r} does not exist in layout {config.layout!r}; "
            f"available: {sorted(config.sectors)}"
        )
    low, _ = config.sectors[name]
    span = sector_span(name, config)
    margin = margin_fraction * span
    theta = math.radians(low + rng.uniform(margin, span - margin))
    return np.array(
        [distance * math.cos(theta), distance * math.sin(theta), 0.0],
        dtype=np.float32,
    )
