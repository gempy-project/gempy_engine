import dataclasses
from dataclasses import dataclass
import numpy as np

@dataclass
class TensorsStructure:
    number_of_points_per_surface: np.ndarray
    len_c_g: np.ndarray
    len_c_gi: np.ndarray
    len_sp: np.ndarray
    len_faults: np.ndarray
    dtype: np.ndarray = np.int32

    _type_conversion_done = False

    def __post_init__(self):
        if self._type_conversion_done: return

        fields = dataclasses.astuple(self)
        numerical_fields = fields[:5]
        dtype_field = fields[5]
        conv_args = [field.astype(dtype_field) for field in numerical_fields]

        self._type_conversion_done = True # Escape loop

        self.__init__(*conv_args, dtype_field)

    def __hash__(self):
        return hash(656)