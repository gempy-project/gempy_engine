from dataclasses import dataclass
from typing import Optional

from . import SurfacePointsInternals, OrientationsInternals, FaultsInternals
from .options import KernelOptions


@dataclass
class SolverInput:
    sp_internal: SurfacePointsInternals
    ori_internal: OrientationsInternals
    options: KernelOptions
    fault_internal: Optional[FaultsInternals] = None
    
    
    
