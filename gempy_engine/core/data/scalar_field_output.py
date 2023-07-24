import dataclasses
from dataclasses import dataclass
from typing import Optional

import numpy as np

from gempy_engine.core.data.exported_fields import ExportedFields
from gempy_engine.core.data.exported_structs import MaskMatrices
from gempy_engine.core.data.grid import Grid
from gempy_engine.core.data.stack_relation_type import StackRelationType


@dataclass
class ScalarFieldOutput:
    weights: np.ndarray
    grid: Grid

    exported_fields: ExportedFields
    mask_components: Optional[MaskMatrices]  # ? DEP
    stack_relation: StackRelationType
    
    values_block: Optional[np.ndarray]  # final values ignoring unconformities
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
    def exported_fields_regular_grid(self):
        scalar_field = self.exported_fields.scalar_field[:self.grid.len_grids[0]]
        gx_field = self.exported_fields.scalar_field[:self.grid.len_grids[0]]
        gy_field = self.exported_fields.scalar_field[:self.grid.len_grids[0]]
        gz_field = self.exported_fields.scalar_field[:self.grid.len_grids[0]]

        return ExportedFields(scalar_field, gx_field, gy_field, gz_field)

    @property
    def values_block_regular_grid(self):
        return self.values_block[:, self.grid.len_grids[0]]
    
    @property
    def mask_components_erode(self) -> MaskMatrices:
        return self._compute_mask_components(self.exported_fields, StackRelationType.ERODE)
    
    @property
    def mask_components_erode_components_onlap(self) -> MaskMatrices:
        return self._compute_mask_components(self.exported_fields, StackRelationType.ONLAP)
    
    @property
    def mask_components_fault(self) -> MaskMatrices:
        return self._compute_mask_components(self.exported_fields, StackRelationType.FAULT)
    
    def get_mask_components_fault_with_thickness(self, thickness: float) -> MaskMatrices:
        return self._compute_mask_components(self.exported_fields, StackRelationType.FAULT, thickness)
    
    def _compute_mask_components(self, exported_fields: ExportedFields, stack_relation: StackRelationType,
                                 fault_thickness: Optional[float] = None) -> MaskMatrices:
        # ! This is how I am setting the stackRelation in gempy
        # is_erosion = self.series.df['BottomRelation'].values[self.non_zero] == 'Erosion'
        # is_onlap = np.roll(self.series.df['BottomRelation'].values[self.non_zero] == 'Onlap', 1)
        # ! if len(is_erosion) != 0:
        # !     is_erosion[-1] = False

        # * These are the default values
        mask_erode = np.ones_like(exported_fields.scalar_field)
        mask_onlap = None  # ! it is the mask of the previous stack (from gempy: mask_matrix[n_series - 1, shift:x_to_interpolate_shape + shift])

        match stack_relation:
            case StackRelationType.ERODE:
                erode_limit_value = exported_fields.scalar_field_at_surface_points.min()
                mask_lith = exported_fields.scalar_field > erode_limit_value
            case StackRelationType.ONLAP:
                onlap_limit_value = exported_fields.scalar_field_at_surface_points.max()
                mask_lith = exported_fields.scalar_field > onlap_limit_value
            case StackRelationType.FAULT:
                # TODO [x] Prototyping thickness for faults
                if fault_thickness is not None:
                    fault_limit_value = exported_fields.scalar_field_at_surface_points.min()
                    thickness_1 = fault_limit_value - fault_thickness
                    thickness_2 = fault_limit_value + fault_thickness

                    f1 = exported_fields.scalar_field > thickness_1
                    f2 = exported_fields.scalar_field < thickness_2

                    exported_fields.scalar_field_at_fault_shell = np.array([thickness_1, thickness_2])
                    mask_lith = f1 * f2
                else:
                    # TODO:  This branch should be like
                    # erode_limit_value = exported_fields.scalar_field_at_surface_points.min()
                    # mask_lith = exported_fields.scalar_field > erode_limit_value

                    mask_lith = np.zeros_like(exported_fields.scalar_field)
            case False | StackRelationType.BASEMENT:
                mask_lith = np.ones_like(exported_fields.scalar_field)
            case _:
                raise ValueError("Stack relation type is not supported")

        return MaskMatrices(mask_lith, None)