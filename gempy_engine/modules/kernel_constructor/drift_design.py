from dataclasses import dataclass
import warnings

import numpy as np

from gempy_engine.core.backend_tensor import BackendTensor
from gempy_engine.core.data.internal_structs import SolverInput
from gempy_engine.core.data.options import KernelOptions


@dataclass
class DriftDesign:
    universal: object
    faults: object
    combined: object
    universal_slice: slice
    fault_slice: slice
    labels: tuple[str, ...]


@dataclass
class DesignMatrixDiagnostics:
    shape: tuple[int, int]
    column_norms: np.ndarray
    zero_columns: tuple[int, ...]
    singular_values: np.ndarray
    rank: int
    relative_smallest_singular_value: float
    condition_number: float
    qr_diagonal: np.ndarray
    dependent_right_vectors: np.ndarray
    normalized_singular_values: np.ndarray
    normalized_rank: int
    normalized_relative_smallest_singular_value: float
    normalized_condition_number: float


@dataclass
class DriftDiagnosticsReport:
    universal: DesignMatrixDiagnostics
    faults: DesignMatrixDiagnostics
    combined: DesignMatrixDiagnostics
    labels: tuple[str, ...]


def build_drift_design(interp_input: SolverInput, options: KernelOptions) -> DriftDesign:
    """Build physical universal and fault observation designs before selector embedding."""
    ori = interp_input.ori_internal
    sp = interp_input.sp_internal
    dimensions = options.number_dimensions
    n_orientations = ori.n_orientations
    orientation_size = ori.n_orientations_tiled
    observation_size = orientation_size + sp.n_points
    n_universal = options.n_uni_eq

    universal = BackendTensor.t.zeros((observation_size, n_universal), dtype=BackendTensor.dtype_obj)
    labels: list[str] = []
    if options.uni_degree != 0:
        coordinate_labels = ("x", "y", "z")[:dimensions]
        labels.extend(coordinate_labels)
        for dimension in range(dimensions):
            universal[n_orientations * dimension:n_orientations * (dimension + 1), dimension] = 1
        universal[orientation_size:, :dimensions] = options.gi_res * (
            sp.ref_surface_points - sp.rest_surface_points
        )

    if options.uni_degree == 2:
        if dimensions != 3:
            raise ValueError("Second-degree drift diagnostics currently require three dimensions")
        positions = ori.orientations.dip_positions
        ref = sp.ref_surface_points
        rest = sp.rest_surface_points
        labels.extend(("x2", "y2", "z2", "xy", "xz", "yz"))

        for dimension in range(3):
            rows = slice(n_orientations * dimension, n_orientations * (dimension + 1))
            universal[rows, 3 + dimension] = 2 * options.gi_res * positions[:, dimension]

        cross_terms = ((0, 1), (0, 2), (1, 2))
        for column_offset, (left, right) in enumerate(cross_terms):
            left_rows = slice(n_orientations * left, n_orientations * (left + 1))
            right_rows = slice(n_orientations * right, n_orientations * (right + 1))
            universal[left_rows, 6 + column_offset] = options.gi_res * positions[:, right]
            universal[right_rows, 6 + column_offset] = options.gi_res * positions[:, left]

        universal[orientation_size:, 3:6] = options.i_res * (ref ** 2 - rest ** 2)
        for column_offset, (left, right) in enumerate(cross_terms):
            universal[orientation_size:, 6 + column_offset] = options.i_res * (
                ref[:, left] * ref[:, right] - rest[:, left] * rest[:, right]
            )

    n_faults = interp_input.fault_internal.n_faults
    if n_faults:
        fault_observations = (
            interp_input.fault_internal.fault_values_ref - interp_input.fault_internal.fault_values_rest
        ).T
        faults = BackendTensor.tfnp.concatenate((
            BackendTensor.t.zeros((orientation_size, n_faults), dtype=BackendTensor.dtype_obj),
            fault_observations,
        ))
    else:
        faults = BackendTensor.t.zeros((observation_size, 0), dtype=BackendTensor.dtype_obj)

    fault_labels = tuple(f"fault_{index}" for index in range(n_faults))
    combined = BackendTensor.tfnp.concatenate((universal, faults), axis=1)
    return DriftDesign(
        universal=universal,
        faults=faults,
        combined=combined,
        universal_slice=slice(0, n_universal),
        fault_slice=slice(n_universal, n_universal + n_faults),
        labels=tuple(labels) + fault_labels,
    )


def analyze_drift_design(design: DriftDesign, rcond: float | None = None) -> DriftDiagnosticsReport:
    return DriftDiagnosticsReport(
        universal=_analyze_matrix(design.universal, rcond),
        faults=_analyze_matrix(design.faults, rcond),
        combined=_analyze_matrix(design.combined, rcond),
        labels=design.labels,
    )


def enforce_rank_policy(
        report: DriftDiagnosticsReport,
        policy: str,
        stack_number: int,
        warning_rcond: float | None = None,
):
    if policy not in ("ignore", "warn", "error"):
        raise ValueError("drift_rank_policy must be 'ignore', 'warn', or 'error'")
    n_columns = report.combined.shape[1]
    rank_deficient = report.combined.normalized_rank != n_columns
    poorly_conditioned = (
        warning_rcond is not None
        and report.combined.normalized_relative_smallest_singular_value < warning_rcond
    )
    if (not rank_deficient and not poorly_conditioned) or policy == "ignore":
        return
    if rank_deficient:
        message = (
            f"Stack {stack_number} drift design is rank deficient: "
            f"normalized rank {report.combined.normalized_rank} for {n_columns} columns."
        )
    else:
        message = (
            f"Stack {stack_number} drift design is poorly conditioned: relative smallest singular value "
            f"{report.combined.normalized_relative_smallest_singular_value:.3e}."
        )
    if policy == "error":
        raise ValueError(message)
    warnings.warn(message, RuntimeWarning, stacklevel=2)


def _analyze_matrix(matrix, rcond: float | None) -> DesignMatrixDiagnostics:
    rows, columns = matrix.shape
    if columns == 0:
        empty = np.empty(0)
        return DesignMatrixDiagnostics(
            shape=(rows, columns),
            column_norms=empty,
            zero_columns=(),
            singular_values=empty,
            rank=0,
            relative_smallest_singular_value=float("nan"),
            condition_number=float("nan"),
            qr_diagonal=empty,
            dependent_right_vectors=np.empty((0, 0)),
            normalized_singular_values=empty,
            normalized_rank=0,
            normalized_relative_smallest_singular_value=float("nan"),
            normalized_condition_number=float("nan"),
        )

    detached = matrix.detach() if hasattr(matrix, "detach") else matrix
    column_norms = BackendTensor.t.linalg.norm(detached, axis=0)
    positive = column_norms > 0
    safe_norms = BackendTensor.t.where(
        positive,
        column_norms,
        BackendTensor.t.ones(column_norms.shape, dtype=column_norms.dtype),
    )
    normalized = detached / safe_norms[None, :]

    raw_values = _svd_summary(detached, rcond)
    normalized_values = _svd_summary(normalized, rcond)
    _, qr_r = BackendTensor.t.linalg.qr(detached, mode="reduced")
    qr_diagonal = BackendTensor.t.abs(BackendTensor.t.diag(qr_r))
    norms_numpy = np.asarray(BackendTensor.t.to_numpy(column_norms))

    return DesignMatrixDiagnostics(
        shape=(rows, columns),
        column_norms=norms_numpy,
        zero_columns=tuple(np.flatnonzero(norms_numpy == 0).tolist()),
        singular_values=raw_values[0],
        rank=raw_values[1],
        relative_smallest_singular_value=raw_values[2],
        condition_number=raw_values[3],
        qr_diagonal=np.asarray(BackendTensor.t.to_numpy(qr_diagonal)),
        dependent_right_vectors=raw_values[4],
        normalized_singular_values=normalized_values[0],
        normalized_rank=normalized_values[1],
        normalized_relative_smallest_singular_value=normalized_values[2],
        normalized_condition_number=normalized_values[3],
    )


def _svd_summary(matrix, rcond):
    _, singular_values, vh = BackendTensor.t.linalg.svd(matrix, full_matrices=True)
    singular_numpy = np.asarray(BackendTensor.t.to_numpy(singular_values))
    vh_numpy = np.asarray(BackendTensor.t.to_numpy(vh))
    if singular_numpy.size == 0:
        return singular_numpy, 0, float("nan"), float("nan"), np.empty((0, matrix.shape[1]))

    epsilon = np.finfo(singular_numpy.dtype).eps
    tolerance = (
        float(rcond) * singular_numpy[0]
        if rcond is not None
        else max(matrix.shape) * epsilon * singular_numpy[0]
    )
    rank = int(np.count_nonzero(singular_numpy > tolerance))
    relative_smallest = float(singular_numpy[-1] / singular_numpy[0]) if singular_numpy[0] else 0.0
    condition = float(singular_numpy[0] / singular_numpy[-1]) if rank == matrix.shape[1] and singular_numpy[-1] else float("inf")
    dependent_vectors = vh_numpy[rank:]
    return singular_numpy, rank, relative_smallest, condition, dependent_vectors
