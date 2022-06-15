from dataclasses import dataclass
from typing import Optional

import numpy as np

from gempy_engine.core.data.exported_structs import CombinedScalarFieldsOutput
from gempy_engine.core.data.scalar_field_output import ScalarFieldOutput


@dataclass(init=True)
class InterpOutput:
    scalar_fields: ScalarFieldOutput
    combined_scalar_field: Optional[CombinedScalarFieldsOutput] = None
    
    @property
    def squeezed_mask_array(self): return self.combined_scalar_field.squeezed_mask_array
    @property
    def final_block(self): return self.combined_scalar_field.final_block
    @property
    def final_exported_fields(self): return self.combined_scalar_field.final_exported_fields
    @property
    def grid_size(self): return self.scalar_fields.grid_size
    @property
    def scalar_field_at_sp(self): return self.scalar_fields.scalar_field_at_sp
    @property
    def exported_fields_regular_grid(self): return self.scalar_fields.exported_fields_regular_grid
    @property
    def values_block_regular_grid(self): return self.scalar_fields.values_block_regular_grid
    @property
    def weights(self): return self.scalar_fields.weights
    @property
    def grid(self): return self.scalar_fields.grid
    @property
    def exported_fields(self): return self.scalar_fields.exported_fields
    @property
    def values_block(self): return self.scalar_fields.values_block
    @property
    def mask_components(self): return self.scalar_fields.mask_components
    @property
    def final_exported_fields(self): return self.combined_scalar_field.final_exported_fields
    @property
    def final_block(self): return self.combined_scalar_field.final_block
    
    @property
    def ids_block_regular_grid(self):
        return np.rint(self.block[:self.grid.len_grids[0]].reshape(self.grid.regular_grid_shape))

    @property
    def ids_block(self) -> np.ndarray:
        return np.rint(self.block[:self.grid.len_grids[0]])

    @ids_block.setter
    def ids_block(self, value):
        self.block[:self.grid.len_grids[0]] = value
    
    @property
    def block(self):
        if self.combined_scalar_field is None:
            return self.values_block
        else:
            return self.combined_scalar_field.final_block