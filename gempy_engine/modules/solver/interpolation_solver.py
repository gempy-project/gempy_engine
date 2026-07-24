import os
import warnings
from dataclasses import dataclass
from enum import Enum, auto

from gempy_engine.config import AvailableBackends
from gempy_engine.core.backend_tensor import BackendTensor
from gempy_engine.core.data.continue_epoch import ContinueEpoch
from gempy_engine.core.data.options import KernelOptions
from gempy_engine.modules.kernel_constructor import kernel_constructor_interface as kernel_constructor
from gempy_engine.modules.kernel_constructor._kernels_assembler import create_cov_kernel
from gempy_engine.modules.kernel_constructor._structs import KernelInput
from gempy_engine.modules.kernel_constructor._vectors_preparation import cov_vectors_preparation
from gempy_engine.modules.kernel_constructor.execution_mode import KernelExecutionMode
from gempy_engine.modules.solver import solver_interface
from gempy_engine.modules.solver.fault_drift_stabilization import stabilize_fault_drift_system
from gempy_engine.modules.solver.linear_system_transform import (
    LinearSystemTransform,
    add_fault_regularization,
    equilibrate_symmetric_system,
    normalized_residual,
)


class InterpolationSolveRoute(Enum):
    DENSE_UNTRANSFORMED = auto()
    DENSE_FAULT_STABILIZED = auto()
    DENSE_RUIZ = auto()
    PYKEOPS_WITH_DENSE_FALLBACK = auto()


@dataclass(frozen=True)
class InterpolationSystem:
    physical_matrix: object
    physical_rhs: object
    transform: LinearSystemTransform | None = None

    @property
    def solve_matrix(self):
        return self.transform.matrix if self.transform is not None else self.physical_matrix

    @property
    def solve_rhs(self):
        return self.transform.rhs if self.transform is not None else self.physical_rhs

    def scale_initial_guess(self, initial_guess):
        return self.transform.scale_initial_guess(initial_guess) if self.transform is not None else initial_guess

    def restore_weights(self, solver_weights):
        return self.transform.restore_weights(solver_weights) if self.transform is not None else solver_weights


@dataclass(frozen=True)
class InterpolationSolveResult:
    route: InterpolationSolveRoute
    execution_mode: KernelExecutionMode
    system: InterpolationSystem
    solver_weights: object
    weights: object
    normalized_residual: float | None
    used_fallback: bool = False


def pykeops_solver_requested() -> bool:
    return os.getenv("PYKEOPS_SOLVER", "False") == "True"


def select_interpolation_solve_route(
        kernel_options: KernelOptions,
        n_faults: int,
        pykeops_requested: bool,
) -> InterpolationSolveRoute:
    method = kernel_options.symmetric_equilibration_method
    match (method, n_faults > 0, pykeops_requested, kernel_options.optimizing_condition_number):
        case ("ruiz", _, True, _):
            _warn_dense_route("symmetric equilibration")
            return InterpolationSolveRoute.DENSE_RUIZ
        case ("ruiz", _, False, _):
            return InterpolationSolveRoute.DENSE_RUIZ
        case ("none", True, True, _):
            _warn_dense_route("fault stabilization")
            return InterpolationSolveRoute.DENSE_FAULT_STABILIZED
        case ("none", True, False, _):
            return InterpolationSolveRoute.DENSE_FAULT_STABILIZED
        case ("none", False, True, True):
            _warn_dense_route("condition-number optimization")
            return InterpolationSolveRoute.DENSE_UNTRANSFORMED
        case ("none", False, True, False):
            if BackendTensor.engine_backend is not AvailableBackends.PYTORCH:
                raise ValueError("The PyKeOps solver requires the PyTorch backend")
            return InterpolationSolveRoute.PYKEOPS_WITH_DENSE_FALLBACK
        case ("none", False, False, _):
            return InterpolationSolveRoute.DENSE_UNTRANSFORMED
        case _:
            raise ValueError(f"Unsupported symmetric equilibration method: {method!r}")


def solve_dense_untransformed(
        interp_input,
        kernel_options: KernelOptions,
        *,
        kernel_data: KernelInput | None = None,
        used_fallback: bool = False,
) -> InterpolationSolveResult:
    kernel_data = kernel_data or cov_vectors_preparation(interp_input, kernel_options)
    system = _build_physical_system(kernel_data, interp_input, kernel_options, KernelExecutionMode.DENSE)
    return _solve_or_raise(
        system,
        interp_input,
        kernel_options,
        InterpolationSolveRoute.DENSE_UNTRANSFORMED,
        KernelExecutionMode.DENSE,
        used_fallback,
    )


def solve_dense_fault_stabilized(interp_input, kernel_options: KernelOptions) -> InterpolationSolveResult:
    n_faults = interp_input.fault_internal.n_faults
    kernel_data = cov_vectors_preparation(interp_input, kernel_options)
    physical_system = _build_physical_system(kernel_data, interp_input, kernel_options, KernelExecutionMode.DENSE)
    transform = stabilize_fault_drift_system(
        covariance=physical_system.physical_matrix,
        b_vector=physical_system.physical_rhs,
        n_faults=n_faults,
        relative_regularization=kernel_options.fault_drift_regularization,
        equilibrate=kernel_options.fault_drift_equilibration,
    )
    system = InterpolationSystem(physical_system.physical_matrix, physical_system.physical_rhs, transform)
    return _solve_or_raise(
        system,
        interp_input,
        kernel_options,
        InterpolationSolveRoute.DENSE_FAULT_STABILIZED,
        KernelExecutionMode.DENSE,
    )


def solve_dense_ruiz(interp_input, kernel_options: KernelOptions) -> InterpolationSolveResult:
    n_faults = interp_input.fault_internal.n_faults
    kernel_data = cov_vectors_preparation(interp_input, kernel_options)
    physical_system = _build_physical_system(kernel_data, interp_input, kernel_options, KernelExecutionMode.DENSE)
    transform = equilibrate_symmetric_system(
        physical_system.physical_matrix,
        physical_system.physical_rhs,
        max_iterations=kernel_options.symmetric_equilibration_max_iterations,
        tolerance=kernel_options.symmetric_equilibration_tolerance,
    )
    transform = add_fault_regularization(
        transform,
        n_faults=n_faults,
        regularization=kernel_options.fault_drift_regularization,
    )
    system = InterpolationSystem(physical_system.physical_matrix, physical_system.physical_rhs, transform)
    return _solve_or_raise(
        system,
        interp_input,
        kernel_options,
        InterpolationSolveRoute.DENSE_RUIZ,
        KernelExecutionMode.DENSE,
    )


def solve_pykeops_with_dense_fallback(interp_input, kernel_options: KernelOptions) -> InterpolationSolveResult:
    tensor_kernel_data = cov_vectors_preparation(interp_input, kernel_options)
    try:
        lazy_kernel_data = tensor_kernel_data.upgrade_tensors()
        lazy_system = _build_physical_system(
            lazy_kernel_data,
            interp_input,
            kernel_options,
            KernelExecutionMode.PYKEOPS,
        )
        result = _solve_system(
            lazy_system,
            interp_input,
            kernel_options,
            InterpolationSolveRoute.PYKEOPS_WITH_DENSE_FALLBACK,
            KernelExecutionMode.PYKEOPS,
        )
    except Exception as error:
        warnings.warn(
            f"PyKeOps interpolation failed; retrying with a dense solve: {error}",
            RuntimeWarning,
            stacklevel=2,
        )
        result = None
    if result is not None:
        return result
    return solve_dense_untransformed(
        interp_input,
        kernel_options,
        kernel_data=tensor_kernel_data,
        used_fallback=True,
    )


def assemble_solve_debug(result: InterpolationSolveResult, kernel_options: KernelOptions) -> dict[str, object]:
    transform = result.system.transform
    return {
        "weights": result.weights,
        "A_matrix": result.system.physical_matrix,
        "b_vector": result.system.physical_rhs,
        "A_matrix_physical": result.system.physical_matrix,
        "b_vector_physical": result.system.physical_rhs,
        "A_matrix_scaled": result.system.solve_matrix,
        "b_vector_scaled": result.system.solve_rhs,
        "scaling_factors": transform.factors if transform is not None else None,
        "weights_scaled": result.solver_weights,
        "weights_physical": result.weights,
        "condition_number_before": kernel_options.condition_number_before,
        "condition_number_after": kernel_options.condition_number_after,
        "normalized_residual": result.normalized_residual,
        "equilibration_iterations": transform.iterations if transform is not None else 0,
        "equilibration_converged": transform.converged if transform is not None else True,
        "solve_route": result.route,
        "used_fallback": result.used_fallback,
    }


def _build_physical_system(
        kernel_data: KernelInput,
        interp_input,
        kernel_options: KernelOptions,
        execution_mode: KernelExecutionMode,
) -> InterpolationSystem:
    matrix = create_cov_kernel(kernel_data, kernel_options, execution_mode=execution_mode)
    rhs = kernel_constructor.yield_b_vector(interp_input.ori_internal, matrix.shape[0])
    return InterpolationSystem(physical_matrix=matrix, physical_rhs=rhs)


def _solve_or_raise(
        system: InterpolationSystem,
        interp_input,
        kernel_options: KernelOptions,
        route: InterpolationSolveRoute,
        execution_mode: KernelExecutionMode,
        used_fallback: bool = False,
) -> InterpolationSolveResult:
    result = _solve_system(system, interp_input, kernel_options, route, execution_mode, used_fallback)
    if result is None:
        raise RuntimeError(f"Interpolation solve failed for route {route.name}")
    return result


def _solve_system(
        system: InterpolationSystem,
        interp_input,
        kernel_options: KernelOptions,
        route: InterpolationSolveRoute,
        execution_mode: KernelExecutionMode,
        used_fallback: bool = False,
) -> InterpolationSolveResult | None:
    if kernel_options.optimizing_condition_number:
        _optimize_nuggets_against_condition_number(system.solve_matrix, interp_input, kernel_options)

    solver_weights = solver_interface.kernel_reduction(
        cov=system.solve_matrix,
        b=system.solve_rhs,
        kernel_options=kernel_options,
        x0=system.scale_initial_guess(interp_input.weights_x0),
        execution_mode=execution_mode,
    )
    if solver_weights is None:
        return None

    weights = system.restore_weights(solver_weights)
    residual = (
        normalized_residual(system.physical_matrix, system.physical_rhs, weights)
        if execution_mode is KernelExecutionMode.DENSE
        else None
    )
    _record_condition_numbers(kernel_options, system, execution_mode)
    return InterpolationSolveResult(
        route=route,
        execution_mode=execution_mode,
        system=system,
        solver_weights=solver_weights,
        weights=weights,
        normalized_residual=residual,
        used_fallback=used_fallback,
    )


def _record_condition_numbers(
        kernel_options: KernelOptions,
        system: InterpolationSystem,
        execution_mode: KernelExecutionMode,
) -> None:
    if not kernel_options.compute_condition_number or execution_mode is KernelExecutionMode.PYKEOPS:
        return
    before = BackendTensor.t.linalg.cond(system.physical_matrix)
    after = BackendTensor.t.linalg.cond(system.solve_matrix)
    kernel_options.condition_number_before = before
    kernel_options.condition_number_after = after
    kernel_options.condition_number = after


def _optimize_nuggets_against_condition_number(matrix, interp_input, kernel_options: KernelOptions) -> None:
    import torch

    cond_number = BackendTensor.t.linalg.cond(matrix)
    nuggets = interp_input.sp_internal.nugget_effect_ref_rest
    l1_reg = torch.norm(nuggets, 2) ** 2
    loss = cond_number - 100_000_000 * l1_reg
    loss.backward()
    kernel_options.condition_number = cond_number
    print(f"Condition number: {cond_number}.")
    raise ContinueEpoch()


def _warn_dense_route(reason: str) -> None:
    warnings.warn(
        f"PyKeOps was requested, but {reason} requires a dense interpolation solve.",
        RuntimeWarning,
        stacklevel=3,
    )
