import warnings
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
    def squeezed_mask_array(self):
        return self.combined_scalar_field.squeezed_mask_array

    @property
    def final_block(self):
        return self.combined_scalar_field.final_block

    @property
    def faults_block(self):
        return self.combined_scalar_field.faults_block

    @property
    def final_exported_fields(self):
        return self.combined_scalar_field.final_exported_fields

    @property
    def grid_size(self):
        return self.scalar_fields.grid_size

    @property
    def scalar_field_at_sp(self):
        return self.scalar_fields.scalar_field_at_sp

    @property
    def exported_fields_regular_grid(self):
        return self.scalar_fields.exported_fields_regular_grid

    @property
    def values_block_regular_grid(self):
        return self.scalar_fields.values_block_regular_grid

    @property
    def weights(self):
        return self.scalar_fields.weights

    @property
    def grid(self):
        return self.scalar_fields.grid

    @property
    def exported_fields(self):
        return self.scalar_fields.exported_fields

    @property
    def values_block(self):
        return self.scalar_fields.values_block

    @property
    def mask_components(self):
        return self.scalar_fields.mask_components_erode

    @property
    def geological_map(self):
        return self.block[self.grid.topography_slice]

    @property
    def sections(self):
        return self.block[self.grid.sections_slice]

    @property
    def custom_grid(self):
        return self.block[self.grid.custom_grid_slice]

    @property
    def ids_block_regular_grid(self):
        return np.rint(self.block[self.grid.regular_grid_slice].reshape(self.grid.regular_grid_shape))

    @property
    def ids_custom_grid(self):
        return np.rint(self.block[self.grid.custom_grid_slice])

    @property
    def ids_block(self) -> np.ndarray:
        return np.rint(self.block[self.grid.regular_grid_slice])

    @ids_block.setter
    def ids_block(self, value):  # ! This is just used for testing or other weird stuff
        warnings.warn("This is just used for testing or other weird stuff")
        self.block[self.grid.regular_grid_slice] = value

    @property
    def block(self):
        if self.combined_scalar_field is None:
            return self.values_block
        else:
            return self.combined_scalar_field.final_block  # * (miguel March 2023) For now faults does not have final block. We will have to add a mask logic for fault blocks first

    @property
    def litho_faults_ids(self):
        if self.combined_scalar_field is None:  # * This in principle is only used for testing
            return self.ids_block

        litho_ids = np.rint(self.block)
        faults_ids = np.rint(self.faults_block)

        # Get the number of unique lithology IDs
        multiplier = len(np.unique(litho_ids))

        # Generate the unique IDs
        unique_ids = litho_ids + faults_ids * multiplier
        return unique_ids
