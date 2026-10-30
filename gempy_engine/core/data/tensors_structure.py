from __future__ import annotations

from dataclasses import dataclass, field
from typing import Type

import numpy as np

from ..backend_tensor import BackendTensor as b, BackendTensor
from ...config import AvailableBackends
from ..utils import cast_type_inplace


# def _cast_type_inplace(struct_data_instance: "TensorStructure"):
#     for key, val in struct_data_instance.__dict__.items():
#         if type(val) != np.ndarray: continue
#         struct_data_instance.__dict__[key] = val.astype(struct_data_instance.dtype)


@dataclass
class TensorsStructure:
    number_of_points_per_surface: np.ndarray
    dtype: Type = np.int32  # ? Isn't this more for options?

    _reference_sp_position: np.ndarray =  field(default_factory=lambda: np.ones(1))

    def __post_init__(self):  # TODO: Move this to init
        counts = self.number_of_points_per_surface
        cast_type_inplace(self, keep_dtype=True)
        # The counts on the host too (zeros removed) when they were given there: the torch backend splits
        # and repeats the surface points with host-built indices (``host_split``) instead of device masks.
        self.host_counts = counts[counts != 0] if isinstance(counts, np.ndarray) else None

        # Set _number_of_points_per_surface_vector
        self.number_of_points_per_surface = self.number_of_points_per_surface[self.number_of_points_per_surface != 0]  # remove 0s
        per_surface_cumsum = BackendTensor.t.cumsum(self.number_of_points_per_surface, 0)#.cumsum()

        self._reference_sp_position = BackendTensor.t.concatenate([BackendTensor.t.array([0]), per_surface_cumsum])[:-1]

    def __hash__(self):
        return hash(656)  # TODO: Make a proper hash

    @classmethod
    def from_tensor_structure_subset(cls, data_descriptor: "InputDataDescriptor", stack_number: int) -> TensorsStructure:
        ts = data_descriptor.tensors_structure
        l0 = data_descriptor.stack_structure.number_of_surfaces_per_stack_vector[stack_number]
        l1 = data_descriptor.stack_structure.number_of_surfaces_per_stack_vector[stack_number + 1]

        n_points_per_surface = ts.number_of_points_per_surface[l0:l1]

        subset = cls(n_points_per_surface, dtype=ts.dtype)
        if getattr(ts, "host_counts", None) is not None:
            subset.host_counts = ts.host_counts[l0:l1]
        return subset

    @property
    def reference_sp_position(self):
        """This is used to find a point on each surface"""
        return self._reference_sp_position

    @property
    def total_number_sp(self):
        return BackendTensor.t.sum(self.number_of_points_per_surface, dtype='int32')

    @property
    def n_surfaces(self):
        return self.number_of_points_per_surface.shape[0]

    def host_split(self):
        """Device row indices of the reference points, the rest points and the reference point repeated for
        each of its surface's rest points, plus each rest point's surface number (int64), or None.

        Built once from ``host_counts`` on the torch backend: gathers with these give exactly what the
        boolean ``partitions_bool`` masks and the repeats by device counts give, without their syncs."""
        if (BackendTensor.engine_backend is not AvailableBackends.PYTORCH
                or not isinstance(getattr(self, "host_counts", None), np.ndarray)):
            return None
        device, split = getattr(self, "_host_split", (None, None))
        if split is None or device != str(BackendTensor.device):
            from ..host_arrays import upload
            counts = self.host_counts.astype(np.int64)
            starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
            rest = np.setdiff1d(np.arange(counts.sum()), starts)  # ascending, as a boolean mask selects
            repeats = counts - 1
            index = upload(np.concatenate((starts, rest, np.repeat(starts, repeats), np.repeat(np.arange(len(counts)), repeats))))
            bounds = np.cumsum([len(starts), len(rest), len(rest)])
            split = (index[:bounds[0]], index[bounds[0]:bounds[1]], index[bounds[1]:bounds[2]], index[bounds[2]:])
            self._host_split = (str(BackendTensor.device), split)
        return split

    @property
    def partitions_bool(self):
        ref_positions = self.reference_sp_position
        res = BackendTensor.t.eye(self.total_number_sp, dtype='int32')[BackendTensor.t.array(ref_positions, dtype="int").reshape(-1)]
        one_hot_ = res.reshape(list(ref_positions.shape) + [self.total_number_sp])
        
        one_hot_ = b.tfnp.array(one_hot_)
            
        partitions = b.tfnp.sum(one_hot_, axis=0, dtype=bool)
        return partitions
