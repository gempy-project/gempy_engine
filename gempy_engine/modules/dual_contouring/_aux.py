def _surface_slicer(surface_i: int, valid_edges_per_surface) -> slice:
    next_surface_edge_idx: int = valid_edges_per_surface[:surface_i + 1].sum()
    if surface_i == 0:
        last_surface_edge_idx = 0
    else:
        last_surface_edge_idx: int = valid_edges_per_surface[:surface_i].sum()
    slice_object: slice = slice(last_surface_edge_idx, next_surface_edge_idx)

    return slice_object
