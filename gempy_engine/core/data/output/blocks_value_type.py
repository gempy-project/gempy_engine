import enum


class ValueType(enum.Enum):
    ids = enum.auto()
    faults_block = enum.auto()
    litho_faults_block = enum.auto()
    
    scalar = enum.auto()
    
    squeeze_mask = enum.auto()
    mask_component = enum.auto()
    values_block = enum.auto()
