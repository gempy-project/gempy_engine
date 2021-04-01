from gempy_engine.config import BackendTensor, AvailableBackends
from gempy_engine.core.data.options import InterpolationOptions
from gempy_engine.core.data.kernel_classes.surface_points import SurfacePointsInternals
from gempy_engine.core.data.kernel_classes.orientations import OrientationsInternals

from ._kernel_constructors import assembly_core_tensor, assembly_ug_tensor, assembly_dips_points_tensor, \
    assembly_dips_ug_coords, assembly_dips_points_coords, assembly_usp_tensor
from ._kernel_selectors import drift_selector, hu_hv_sel
from ._structs import KernelInput


def _vectors_preparation(sp_: SurfacePointsInternals, ori_: OrientationsInternals,
                         options: InterpolationOptions) -> KernelInput:
    cov_size = ori_.n_orientations_tiled + sp_.n_points + options.n_uni_eq

    dips_ref_coord = assembly_dips_points_tensor(ori_.dip_positions_tiled, sp_.ref_surface_points, options)
    dips_rest_coord = assembly_dips_points_tensor(ori_.dip_positions_tiled, sp_.rest_surface_points, options)

    # When we create que core covariance this are the repeated since the distance are with themselves
    orientations_sp_matrices = assembly_core_tensor(dips_ref_coord, dips_ref_coord, dips_rest_coord, dips_rest_coord)

    # Universal
    dips_ug_d1, dips_ug_d2 = assembly_dips_ug_coords(ori_, sp_.n_points, options)
    dips_ug = assembly_ug_tensor(dips_ug_d1, dips_ug_d1, dips_ug_d2, dips_ug_d1)

    dips_ref_d1, dips_ref_d2a, dips_ref_d2b = assembly_dips_points_coords(sp_.ref_surface_points,
                                                                          ori_.n_orientations_tiled, options)
    dips_rest_d1, dips_rest_d2a, dips_rest_d2b = assembly_dips_points_coords(sp_.rest_surface_points,
                                                                                   ori_.n_orientations_tiled, options)

    dips_ref_ui = assembly_usp_tensor(dips_ref_d1, dips_ref_d1, dips_ref_d2a, dips_ref_d2a, dips_ref_d2b, dips_ref_d2b)
    dips_rest_ui = assembly_usp_tensor(dips_rest_d1, dips_rest_d1, dips_rest_d2a, dips_rest_d2a, dips_rest_d2b,
                                       dips_rest_d2b)

    # Selectors :
    # TODO: This are going to be the same? no actually I do not think so!
    cartesian_selector = hu_hv_sel(ori_.n_orientations, options.number_dimensions, sp_.n_points, cov_size)
    drift_selection = drift_selector(cov_size, options.n_uni_eq)

    kernel_input_args = [orientations_sp_matrices, cartesian_selector, dips_ug, dips_ref_ui, dips_rest_ui,
                         drift_selection]
    # Prepare Return
    ki = _new_kernel_input(kernel_input_args)

    return ki


def _new_kernel_input(kernel_input_args) -> KernelInput:
    def _upgrade_kernel_input_to_keops_tensor(args):
        from pykeops.numpy import LazyTensor
        import dataclasses

        for arg in args:
            new_args = [LazyTensor(i.astype('float32')) for i in dataclasses.astuple(arg)]
            arg.__init__(*new_args)

    if BackendTensor.engine_backend == AvailableBackends.numpy and BackendTensor.pykeops_enabled:
        _upgrade_kernel_input_to_keops_tensor(kernel_input_args)

    ki = KernelInput(*kernel_input_args)
    return ki
