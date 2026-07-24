from enum import Enum, auto


class KernelExecutionMode(Enum):
    DENSE = auto()
    PYKEOPS = auto()
