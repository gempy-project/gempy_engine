import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np

from gempy_engine.config import AvailableBackends
from gempy_engine.core.backend_tensor import BackendTensor
from gempy_engine.core.data.exported_structs import CombinedScalarFieldsOutput
from gempy_engine.core.data.output.blocks_value_type import ValueType
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
    def exported_fields_dense_grid(self):
        return self.scalar_fields.exported_fields_dense_grid

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
    def custom_grid_values(self):
        return self.block[self.grid.custom_grid_slice]
    
    @property
    def geophysics_grid_values(self):
        return self.block[self.grid.geophysics_grid_slice]
    
    @property
    def cornersGrid_values(self):
        return self.block[self.grid.corners_grid_slice]

    @property
    def ids_cornersGrid(self):
        return BackendTensor.t.rint(self.block[self.grid.corners_grid_slice])

    @property
    def ids_geophysics_grid(self):
        return BackendTensor.t.rint(self.block[self.grid.geophysics_grid_slice])

    @property
    def ids_block_octree_grid(self):
        block = self.block.reshape(-1)
        return BackendTensor.t.rint(block[self.grid.octree_grid_slice].reshape(self.grid.octree_grid_shape.tolist()))
    
    @property
    def ids_block_dense_grid(self):
        block = self.block.reshape(-1)
        return BackendTensor.t.rint(block[self.grid.dense_grid_slice].reshape(self.grid.dense_grid_shape.tolist()))

    @property
    def ids_custom_grid(self):
        return BackendTensor.t.rint(self.block[self.grid.custom_grid_slice])

    @property
    def ids_block(self) -> np.ndarray:
        return BackendTensor.t.rint(self.block[self.grid.octree_grid_slice])

    @ids_block.setter
    def ids_block(self, value):  # ! This is just used for testing or other weird stuff
        warnings.warn("This is just used for testing or other weird stuff")
        self.block[self.grid.octree_grid_slice] = value

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

        litho_ids = BackendTensor.t.rint(self.block)
        faults_ids = BackendTensor.t.rint(self.faults_block)

        # Get the number of unique lithology IDs
        multiplier = len(BackendTensor.t.unique(litho_ids))
    
        # Generate the unique IDs
        unique_ids = litho_ids + faults_ids * multiplier
        return unique_ids

    @property
    def litho_faults_ids_corners_grid(self):
        if self.combined_scalar_field is None:  # * This in principle is only used for testing
            return self.ids_cornersGrid

        litho_ids = BackendTensor.t.rint(self.block[self.grid.corners_grid_slice])
        faults_ids = BackendTensor.t.rint(self.faults_block[self.grid.corners_grid_slice])

        # Get the number of unique lithology IDs
        multiplier = len(BackendTensor.t.unique(litho_ids))

        # Generate the unique IDs
        unique_ids = litho_ids + faults_ids * multiplier
        return unique_ids
    
    def get_block_from_value_type(self, value_type: ValueType, slice_: slice):
        match value_type:
            case ValueType.ids:
                block = self.final_block
            case ValueType.faults_block:
                block = self.faults_block
            case ValueType.litho_faults_block:
                block = self.litho_faults_ids
            case ValueType.values_block:
                block = self.values_block[0]
            case ValueType.scalar:
                block = self.exported_fields.scalar_field
            case ValueType.squeeze_mask:
                block = self.squeezed_mask_array
            case ValueType.mask_component:
                block = self.mask_components
            case _:
                raise ValueError("ValueType not supported.")
        
        match (BackendTensor.engine_backend):
            case AvailableBackends.PYTORCH:
                block = BackendTensor.t.to_numpy(block)
                # block = block.to_numpy()

        return block[slice_]
