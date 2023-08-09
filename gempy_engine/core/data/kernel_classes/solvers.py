from enum import Enum, auto


class Solvers(Enum):
    DEFAULT = auto()
    PYKEOPS_CG = auto()
    SCIPY_CG = auto()
    GMRES = auto()

    