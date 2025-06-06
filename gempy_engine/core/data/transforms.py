﻿import pprint
import warnings
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

import numpy as np
from typing_extensions import Annotated

from .encoders.converters import numpy_array_short_validator


class TransformOpsOrder(Enum):
    TRS = auto()  # * The order of the transformations is: scale, rotation, translation
    SRT = auto()  # * The order of the transformations is: translation, rotation, scale


class GlobalAnisotropy(Enum):
    CUBE = auto()  # * Transform data to be as close as possible to a cube
    NONE = auto()  # * Do not transform data
    MANUAL = auto()  # * Use the user defined transform


@dataclass
class Transform:
    position: Annotated[np.ndarray, numpy_array_short_validator]
    rotation: Annotated[np.ndarray, numpy_array_short_validator]
    scale: Annotated[np.ndarray, numpy_array_short_validator]

    _is_default_transform: bool = False
    _cached_pivot: Annotated[np.ndarray, numpy_array_short_validator] | None = None

    def __repr__(self):
        return pprint.pformat(self.__dict__)

    def __post_init__(self):
        assert self.position.shape == (3,)
        assert self.rotation.shape == (3,)
        assert self.scale.shape == (3,)

    def __add__(self, other):
        if not isinstance(other, Transform):
            raise ValueError("Both objects must be an instance of Transform")

        new_position = self.position + other.position
        new_rotation = self.rotation + other.rotation
        new_scale = self.scale * other.scale

        return Transform(new_position, new_rotation, new_scale)

    @classmethod
    def init_neutral(cls):
        t = cls(position=np.zeros(3), rotation=np.zeros(3), scale=np.ones(3))
        t._cached_pivot = np.zeros(3)
        return t

    @classmethod
    def from_matrix(cls, matrix: np.ndarray):
        assert matrix.shape == (4, 4)
        position = matrix[:3, 3]
        rotation_radians = np.array([
                np.arctan2(matrix[2, 1], matrix[2, 2]),
                np.arctan2(-matrix[2, 0], np.sqrt(matrix[2, 1] ** 2 + matrix[2, 2] ** 2)),
                np.arctan2(matrix[1, 0], matrix[0, 0])
        ])
        rotation_degrees = np.degrees(rotation_radians)
        scale = np.array([
                np.linalg.norm(matrix[0, :3]),
                np.linalg.norm(matrix[1, :3]),
                np.linalg.norm(matrix[2, :3])
        ])
        return cls(position, rotation_degrees, scale)

    @property
    def cached_pivot(self):
        return self._cached_pivot

    @cached_pivot.setter
    def cached_pivot(self, pivot: list):
        self._cached_pivot = np.array(pivot)

    @classmethod
    def from_input_points(cls, surface_points: 'gempy.data.SurfacePointsTable', orientations: 'gempy.data.OrientationsTable') -> 'Transform':
        input_points_xyz = np.concatenate([surface_points.xyz, orientations.xyz], axis=0)
        if input_points_xyz.shape[0] == 0:
            transform = cls(position=np.zeros(3), rotation=np.zeros(3), scale=np.ones(3))
            transform._is_default_transform = True
            return transform

        max_coord = np.max(input_points_xyz, axis=0)
        min_coord = np.min(input_points_xyz, axis=0)

        # Compute the range for each dimension
        range_coord = 2 * (max_coord - min_coord)

        # Avoid division by zero; replace zero with a small number
        range_coord = np.where(range_coord == 0, 1e-10, range_coord)

        # The scaling factor for each dimension is the inverse of its range
        scaling_factors = 1 / range_coord

        # ! Be careful with toy models
        center: np.ndarray = (max_coord + min_coord) / 2
        return cls(
            position=-center,
            rotation=np.zeros(3),
            scale=cls._adjust_scale_to_limit_ratio(
                s=scaling_factors,
                anisotropic_limit=np.array([1, 1, 1])  # ! Increase this number to auto anisotropy
            )
        )

    def apply_anisotropy(self, anisotropy_type: GlobalAnisotropy, anisotropy_limit: Optional[np.ndarray] = None):
        if anisotropy_type == GlobalAnisotropy.CUBE:
            warnings.warn(
                message="Interpolation is being done with the default transform. "
                        "If you do not know what you are doing you should probably call GeoModel.update_transform() first.",
                category=RuntimeWarning
            )
        elif anisotropy_type == GlobalAnisotropy.NONE:
            self.scale = self._adjust_scale_to_limit_ratio(
                s=self.scale,
                anisotropic_limit=np.array([1, 1, 1])  # ! Increase this number to auto anisotropy
            )
        elif anisotropy_type == GlobalAnisotropy.MANUAL and anisotropy_limit is not None:
            self.scale = self._adjust_scale_to_limit_ratio(
                s=self.scale,
                anisotropic_limit=anisotropy_limit
            )
        else:
            raise NotImplementedError

    @staticmethod
    def _adjust_scale_to_limit_ratio(s, anisotropic_limit=np.array([10, 10, 10])):
        # Calculate the ratios
        ratios = [
                s[0] / s[1], s[0] / s[2],
                s[1] / s[0], s[1] / s[2],
                s[2] / s[0], s[2] / s[1]
        ]

        # Adjust the scales based on the index of the max ratio
        if ratios[0] > anisotropic_limit[0]:
            s[0] = s[1] * anisotropic_limit[0]
        if ratios[1] > anisotropic_limit[0]:
            s[0] = s[2] * anisotropic_limit[0]

        if ratios[2] > anisotropic_limit[1]:
            s[1] = s[0] * anisotropic_limit[1]
        if ratios[3] > anisotropic_limit[1]:
            s[1] = s[2] * anisotropic_limit[1]

        if ratios[4] > anisotropic_limit[2]:
            s[2] = s[0] * anisotropic_limit[2]
        if ratios[5] > anisotropic_limit[2]:
            s[2] = s[1] * anisotropic_limit[2]

        return s

    @staticmethod
    def _max_scale_ratio(s):
        ratios = [
                s[0] / s[1], s[0] / s[2],
                s[1] / s[0], s[1] / s[2],
                s[2] / s[0], s[2] / s[1]
        ]
        return max(ratios)

    @property
    def isometric_scale(self):
        # TODO: double check how was done in old gempy
        return 1 / np.mean(self.scale)

    def get_transform_matrix(self, transform_type: TransformOpsOrder = TransformOpsOrder.SRT) -> np.ndarray:
        T = np.eye(4)
        R = np.eye(4)
        S = np.eye(4)

        T[:3, 3] = self.position

        rx, ry, rz = np.radians(self.rotation)
        Rx = np.array(
            [[1, 0, 0],
             [0, np.cos(rx), -np.sin(rx)],
             [0, np.sin(rx), np.cos(rx)]]
        )

        Ry = np.array(
            [[np.cos(ry), 0, np.sin(ry)],
             [0, 1, 0],
             [-np.sin(ry), 0, np.cos(ry)]]
        )

        Rz = np.array(
            [[np.cos(rz), -np.sin(rz), 0],
             [np.sin(rz), np.cos(rz), 0],
             [0, 0, 1]]
        )

        R[:3, :3] = Rx @ Ry @ Rz
        S[:3, :3] = np.diag(self.scale)

        match transform_type:
            case TransformOpsOrder.TRS:
                return T @ R @ S
            case TransformOpsOrder.SRT:
                return S @ R @ T
            case _:
                raise NotImplementedError(f"Transform type {transform_type} not implemented")

    def apply(self, points: np.ndarray, transform_op_order: TransformOpsOrder = TransformOpsOrder.SRT):
        # * NOTE: to compare with legacy we would have to add 0.5 to the coords
        assert points.shape[1] == 3
        if self._is_default_transform:
            warnings.warn(
                message="Interpolation is being done with the default transform. "
                        "If you do not know what you are doing you should probably call GeoModel.update_transform() first.",
                category=RuntimeWarning
            )

        homogeneous_points = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
        matrix = self.get_transform_matrix(transform_op_order)
        transformed_points = (matrix @ homogeneous_points.T).T
        return transformed_points[:, :3]

    def scale_points(self, points: np.ndarray):
        return points * self.scale

    def apply_inverse(self, points: np.ndarray, transform_op_order: TransformOpsOrder = TransformOpsOrder.SRT):
        # * NOTE: to compare with legacy we would have to add 0.5 to the coords
        assert points.shape[1] == 3
        homogeneous_points = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
        matrix = self.get_transform_matrix(transform_op_order)
        inv = np.linalg.inv(matrix)
        transformed_points = (inv @ homogeneous_points.T).T
        return transformed_points[:, :3]

    def apply_with_cached_pivot(self, points: np.ndarray, transform_op_order: TransformOpsOrder = TransformOpsOrder.SRT):
        if self._cached_pivot is None:
            raise ValueError("A pivot must be set before calling this method")
        return self.apply_with_pivot(points, self._cached_pivot, transform_op_order)

    def apply_inverse_with_cached_pivot(self, points: np.ndarray, transform_op_order: TransformOpsOrder = TransformOpsOrder.SRT):
        if self._cached_pivot is None:
            raise ValueError("A pivot must be set before calling this method")
        return self.apply_inverse_with_pivot(points, self._cached_pivot, transform_op_order)

    def apply_with_pivot(self, points: np.ndarray, pivot: np.ndarray,
                         transform_op_order: TransformOpsOrder = TransformOpsOrder.SRT):
        """These are used for ellipsoids for finite faults"""
        assert points.shape[1] == 3
        if self._is_default_transform:
            warnings.warn(
                message="Interpolation is being done with the default transform. "
                        "If you do not know what you are doing you should probably call GeoModel.update_transform() first.",
                category=RuntimeWarning
            )

        # Translation matrices to and from the pivot
        T_to_origin = self._translation_matrix(-pivot[0], -pivot[1], -pivot[2])
        T_back = self._translation_matrix(*pivot)

        # Construct the transformation matrix with the pivot
        M = T_back @ self.get_transform_matrix(transform_op_order) @ T_to_origin

        homogeneous_points = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
        transformed_points = (M @ homogeneous_points.T).T
        return transformed_points[:, :3]

    def apply_inverse_with_pivot(self, points: np.ndarray, pivot: np.ndarray,
                                 transform_op_order: TransformOpsOrder = TransformOpsOrder.SRT):
        assert points.shape[1] == 3

        # Translation matrices to and from the pivot
        T_to_origin = self._translation_matrix(-pivot[0], -pivot[1], -pivot[2])
        T_back = self._translation_matrix(*pivot)

        # Construct the inverse transformation matrix with the pivot
        M_inv = np.linalg.inv(T_back @ self.get_transform_matrix(transform_op_order) @ T_to_origin)

        homogeneous_points = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
        transformed_points = (M_inv @ homogeneous_points.T).T
        return transformed_points[:, :3]

    @staticmethod
    def _translation_matrix(tx, ty, tz):
        return np.array([
                [1, 0, 0, tx],
                [0, 1, 0, ty],
                [0, 0, 1, tz],
                [0, 0, 0, 1]
        ])

    def transform_gradient(self, gradients: np.ndarray, transform_op_order: TransformOpsOrder = TransformOpsOrder.SRT,
                           preserve_magnitude: bool = True) -> np.ndarray:
        assert gradients.shape[1] == 3

        # Extract the 3x3 upper-left section of the transformation matrix
        transformation_3x3 = self.get_transform_matrix(transform_op_order)[:3, :3]

        # Compute the inverse transpose of this 3x3 matrix
        inv_trans_3x3 = np.linalg.inv(transformation_3x3).T

        # Multiply the gradients by this inverse transpose matrix
        transformed_gradients = (inv_trans_3x3 @ gradients.T).T

        if preserve_magnitude:
            # Compute the magnitude of the original gradients
            gradient_magnitudes = np.linalg.norm(gradients, axis=1)

            # Compute the magnitude of the transformed gradients
            transformed_gradient_magnitudes = np.linalg.norm(transformed_gradients, axis=1)

            # Compute the ratio between the two magnitudes
            magnitude_ratio = transformed_gradient_magnitudes / gradient_magnitudes

            # Multiply the transformed gradients by this ratio
            transformed_gradients /= magnitude_ratio[:, None]

        return transformed_gradients
