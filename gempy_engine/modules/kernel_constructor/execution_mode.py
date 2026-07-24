from enum import Enum, auto


class KernelExecutionMode(Enum):
    DENSE = auto()
    SYMBOLIC = auto()
