

def export_scalar(ei: ExportInput, hu_x0, drift, kernel_1st,
                  interpolation_options,
                  ):
    a = interpolation_options.range
    r_dip_x0 = tfnp.sqrt(((ei.dips_i - ei.grid_j) ** 2).sum(-1))
    k_p_grad = kernel_1st(r_dip_x0, a)
    sigma_0_grad_interface = hu_x0 * k_p_grad
    r_ref_grid = tfnp.sqrt(((ei.ref_i - ei.grid_j) ** 2).sum(-1))
    r_rest_grid = tfnp.sqrt(((ei.rest_i - ei.grid_j) ** 2).sum(-1))

    k_ref_x0 = kernel_1st(r_ref_grid, a)
    k_rest_x0 = kernel_1st(r_rest_grid, a)
    sigma_0_interf = k_rest_x0 - k_ref_x0

    return sigma_0_grad_interface + sigma_0_interf + drift

    #
    # a = interpolation_options.range
    #
    # if pykeops_imported is True:
    #     r_dip_x0 = (((ei.dips_i - ei.grid_j) ** 2).sum(-1)).sqrt()
    # else:
    #     r_dip_x0 = tfnp.sqrt(((ei.dips_i - ei.grid_j) ** 2).sum(-1))
    #
    # k_p_grad = kernel_1st(r_dip_x0, a)s
    #
    # sigma_0_grad_interface = hu_x0 * k_p_grad
    #
    # if pykeops_imported is True:
    #     r_ref_grid = (((ei.ref_i - ei.grid_j) ** 2).sum(-1)).sqrt()
    #     r_rest_grid = (((ei.rest_i - ei.grid_j) ** 2).sum(-1)).sqrt()
    #
    # else:
    #     r_ref_grid = tfnp.sqrt(((ei.ref_i - ei.grid_j) ** 2).sum(-1))
    #     r_rest_grid = tfnp.sqrt(((ei.rest_i - ei.grid_j) ** 2).sum(-1))
    #
    # k_ref_x0 = kernel_1st(r_ref_grid, a)
    # k_rest_x0 = kernel_1st(r_rest_grid, a)
    #
    # sigma_0_interf = k_rest_x0 - k_ref_x0
    #
    # return sigma_0_grad_interface + sigma_0_interf + drift
