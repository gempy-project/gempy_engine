import dataclasses
from dataclasses import dataclass
from typing import Optional

import numpy as np

from gempy_engine.core.backend_tensor import BackendTensor
from gempy_engine.core.data.exported_fields import ExportedFields
from gempy_engine.core.data.engine_grid import EngineGrid
from gempy_engine.core.data.stack_relation_type import StackRelationType


@dataclass
class ScalarFieldOutput:
    weights: np.ndarray
    grid: EngineGrid

    exported_fields: ExportedFields
    stack_relation: StackRelationType

    values_block: Optional[np.ndarray]  #: Final values ignoring unconformities
    _values_block: Optional[np.ndarray] = dataclasses.field(init=False, repr=False)

    @property
    def values_block(self) -> Optional[np.ndarray]:
        return self._values_block[:, :self.grid_size]

    @values_block.setter
    def values_block(self, value: np.ndarray):
        self._values_block = value

    @property
    def values_on_all_xyz(self) -> np.ndarray:
        return self._values_block

    @property
    def grid_size(self):
        return self.exported_fields._grid_size

    @property
    def n_points_per_surface(self):
        return self.exported_fields._n_points_per_surface

    @property
    def scalar_field_at_sp(self):
        return self.exported_fields.scalar_field_at_surface_points

    @property
    def exported_fields_dense_grid(self):
        slicer = self.grid.dense_grid_slice
        scalar_field = self.exported_fields.scalar_field[slicer]
        gx_field = self.exported_fields.scalar_field[slicer]
        gy_field = self.exported_fields.scalar_field[slicer]
        gz_field = self.exported_fields.scalar_field[slicer]

        return ExportedFields(scalar_field, gx_field, gy_field, gz_field)

    @property
    def values_block_regular_grid(self):
        return self.values_block[:, self.grid.len_grids[0]]

    @property
    def mask_components_erode(self) -> np.ndarray:
        return self._compute_mask_components(self.exported_fields, StackRelationType.ERODE)

    @property
    def mask_components_erode_components_onlap(self) -> np.ndarray:
        return self._compute_mask_components(self.exported_fields, StackRelationType.ONLAP)

    @property
    def mask_components_basement(self) -> np.ndarray:
        return self._compute_mask_components(self.exported_fields, StackRelationType.BASEMENT)

    @property
    def mask_components_fault(self) -> np.ndarray:
        return self._compute_mask_components(self.exported_fields, StackRelationType.FAULT)

    def get_mask_components_fault_with_thickness(self, thickness: float) -> np.ndarray:
        return self._compute_mask_components(self.exported_fields, StackRelationType.FAULT, thickness)

    @staticmethod
    def _compute_mask_components(exported_fields: ExportedFields, stack_relation: StackRelationType,
                                 fault_thickness: Optional[float] = None) -> np.ndarray:
        match stack_relation:
            case StackRelationType.ERODE:
                erode_limit_value = exported_fields.scalar_field_at_surface_points.min()
                mask_array = exported_fields.scalar_field > erode_limit_value
            case StackRelationType.ONLAP:
                onlap_limit_value = exported_fields.scalar_field_at_surface_points.max()
                mask_array = exported_fields.scalar_field > onlap_limit_value
            case StackRelationType.FAULT:
                # TODO [x] Prototyping thickness for faults
                if fault_thickness is not None:
                    fault_limit_value = exported_fields.scalar_field_at_surface_points.min()
                    thickness_1 = fault_limit_value - fault_thickness
                    thickness_2 = fault_limit_value + fault_thickness

                    f1 = exported_fields.scalar_field > thickness_1
                    f2 = exported_fields.scalar_field < thickness_2

                    exported_fields.scalar_field_at_fault_shell = BackendTensor.t.array([thickness_1, thickness_2])
                    mask_array = f1 * f2
                else:
                    # TODO:  This branch should be like
                    # ? Is the commented out for finite faults?
                    # erode_limit_value = exported_fields.scalar_field_at_surface_points.min()
                    # mask_lith = exported_fields.scalar_field > erode_limit_value

                    mask_array = BackendTensor.t.zeros_like(exported_fields.scalar_field)
            case False | StackRelationType.BASEMENT:
                mask_array = BackendTensor.t.ones_like(exported_fields.scalar_field)
            case _:
                raise ValueError("Stack relation type is not supported")

        return mask_array
