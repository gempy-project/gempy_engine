from __future__ import annotations

import math
from enum import Enum
from typing import Any

import numpy as np
from pydantic import ConfigDict, field_validator, model_validator
from pydantic.dataclasses import dataclass


class TaperType(str, Enum):
    CUBIC = "cubic"
    QUADRATIC = "quadratic"
    SPLINE = "spline"


Radius = float | tuple[float, float]
SplineControlPoints = tuple[tuple[float, float], ...]


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class FiniteFault:
    """Serializable definition of an approximately planar finite fault."""

    center: tuple[float, float, float]
    strike_radius: Radius = 1.0
    dip_radius: Radius = 1.0
    taper: TaperType = TaperType.CUBIC
    rotation_deg: float = 0.0
    spline_control_points: SplineControlPoints | None = None

    @field_validator("center", "strike_radius", "dip_radius", "spline_control_points", mode="before")
    @classmethod
    def _convert_numpy_arrays(cls, value: Any) -> Any:
        return value.tolist() if isinstance(value, np.ndarray) else value

    @field_validator("center")
    @classmethod
    def _validate_center(cls, value: tuple[float, float, float]) -> tuple[float, float, float]:
        if not all(math.isfinite(component) for component in value):
            raise ValueError("center coordinates must be finite")
        return value

    @field_validator("strike_radius", "dip_radius")
    @classmethod
    def _validate_radius(cls, value: Radius) -> Radius:
        radii = value if isinstance(value, tuple) else (value,)
        if not all(math.isfinite(radius) and radius > 0 for radius in radii):
            raise ValueError("radii must be finite and greater than zero")
        return value

    @field_validator("rotation_deg")
    @classmethod
    def _validate_rotation(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("rotation_deg must be finite")
        return value

    @model_validator(mode="after")
    def _validate_spline_control_points(self) -> "FiniteFault":
        points = self.spline_control_points
        if points is None:
            return self
        if self.taper is not TaperType.SPLINE:
            raise ValueError("spline_control_points require a spline taper")
        if len(points) < 2:
            raise ValueError("spline_control_points require at least two points")

        distances = tuple(point[0] for point in points)
        multipliers = tuple(point[1] for point in points)
        if distances[0] != 0.0 or distances[-1] != 1.0:
            raise ValueError("spline distances must start at 0 and end at 1")
        if not all(math.isfinite(value) for point in points for value in point):
            raise ValueError("spline control points must be finite")
        if not all(left < right for left, right in zip(distances, distances[1:])):
            raise ValueError("spline distances must be strictly increasing")
        if not all(0.0 <= multiplier <= 1.0 for multiplier in multipliers):
            raise ValueError("spline multipliers must be between 0 and 1")
        return self

    def calculate_slip(self, points: np.ndarray, normal: np.ndarray) -> np.ndarray:
        """Calculate the prototype slip multiplier for the supplied points."""
        from gempy_engine.modules.faults.finite_faults import calculate_slip

        return calculate_slip(self, points=points, normal=normal)
