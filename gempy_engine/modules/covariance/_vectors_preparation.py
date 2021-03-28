from gempy_engine.config import BackendConf, AvailableBackends
from gempy_engine.core.data.options import InterpolationOptions
from gempy_engine.modules.covariance._kernel_constructors import assembly_dips_surface_points_coord_matrix, hu_hv_sel, \
    input_ug, input_usp, drift_selector
from gempy_engine.modules.covariance._structs import SurfacePointsInternals, OrientationsInternals, KernelInput


def _vectors_preparation(sp_internals: SurfacePointsInternals, ori_internals: OrientationsInternals,
                         interpolation_options: InterpolationOptions) -> KernelInput:
    cov_size = ori_internals.n_orientations_tiled + sp_internals.n_points + interpolation_options.n_uni_eq

    orientations_sp_matrices = assembly_dips_surface_points_coord_matrix(ori_internals.dip_positions_tiled,
                                                                         sp_internals, interpolation_options)

    cartesian_selector = hu_hv_sel(ori_internals.n_orientations, interpolation_options.number_dimensions,
                                   sp_internals.n_points, cov_size)

    dips_ug = input_ug(ori_internals, sp_internals.n_points, interpolation_options)

    # Universal
    dips_ref_ui = input_usp(sp_internals.ref_surface_points, ori_internals.n_orientations_tiled, interpolation_options)

    dips_rest_ui = input_usp(sp_internals.rest_surface_points, ori_internals.n_orientations_tiled,
                             interpolation_options)

    drift_selection= drift_selector(cov_size, interpolation_options.n_uni_eq)

    # Prepare Return
    if BackendConf.engine_backend == AvailableBackends.numpy and BackendConf.pykeops_enabled:
        _upgrade_kernel_input_to_keops_tensor(cartesian_selector, dips_ref_ui, dips_rest_ui, dips_ug, drift_selection,
                                              orientations_sp_matrices)

    ki = KernelInput(orientations_sp_matrices, cartesian_selector, dips_ug, dips_ref_ui,
                     dips_rest_ui, drift_selection)

    return ki


def _upgrade_kernel_input_to_keops_tensor(cartesian_selector, dips_ref_ui, dips_rest_ui, dips_ug, drift_selection,
                                          orientations_sp_matrices):
    from pykeops.numpy import LazyTensor
    args = [orientations_sp_matrices, cartesian_selector, dips_ug, dips_ref_ui, dips_rest_ui, drift_selection]
    for arg in args:
        new_args = [LazyTensor(i.astype('float32')) for i in dataclasses.astuple(arg)]
        args.__init__(new_args)
