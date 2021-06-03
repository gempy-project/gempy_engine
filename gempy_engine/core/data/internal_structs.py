from dataclasses import dataclass

from . import SurfacePointsInternals, OrientationsInternals, InterpolationOptions


@dataclass
class SolverInput:
    sp_internal: SurfacePointsInternals
    ori_internal: OrientationsInternals
    options: InterpolationOptions