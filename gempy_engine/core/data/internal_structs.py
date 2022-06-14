from dataclasses import dataclass

from . import SurfacePointsInternals, OrientationsInternals
from .options import KernelOptions


@dataclass
class SolverInput:
    sp_internal: SurfacePointsInternals
    ori_internal: OrientationsInternals
    options: KernelOptions
