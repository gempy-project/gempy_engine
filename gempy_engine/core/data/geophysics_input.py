from dataclasses import dataclass

from ..backend_tensor import BackendTensor


@dataclass
class GeophysicsInput():
    tz: BackendTensor.t
    densities: BackendTensor.t