import concurrent.futures
from collections import defaultdict
from typing import List

import numpy as np

from gempy_engine.core.backend_tensor import BackendTensor
from gempy_engine.core.data.dual_contouring_mesh import DualContouringMesh
from gempy_engine.modules.dual_contouring.apply_mesh_modifications import remove_triangles_in_voxels


def average_overlapping_vertices(
        all_meshes: List[DualContouringMesh],
        left_right_per_mesh: List[np.ndarray],
        base_number: tuple[int, int, int],
        surface_to_stack: List[int] = None,
        stacks_structure: 'StacksStructure' = None
) -> None:
    """Vertex sharing at overlap voxels.
    
    Three cases based on the relationship between overlapping surfaces:
    
    1. **Same stack**: no vertex sharing at all (surfaces within the same
       geological stack should not modify each other's vertices).
    2. **Fault→layer**: the layer takes the fault's vertex directly (no
       averaging).  The fault surface keeps its own vertex untouched.
    3. **Erosion/onlap (different stacks, non-fault)**: vertices are averaged
       so both surfaces meet at a shared position for watertightness.
    """
    from ...modules.dual_contouring._find_vertex_overlap import _generate_voxel_codes

    n = len(all_meshes)
    if n < 2:
        return

    codes = _generate_voxel_codes(left_right_per_mesh, base_number)

    # --- Determine directed fault pairs and same-stack pairs ---------------
    fault_directed_pairs: set = set()  # (fault_surf, layer_surf)
    same_stack_pairs: set = set()  # unordered pairs within same stack

    if surface_to_stack is not None:
        # Build same-stack pairs
        _stack_to_surfs: dict = defaultdict(list)
        for si, sk in enumerate(surface_to_stack):
            _stack_to_surfs[sk].append(si)

        for surfs in _stack_to_surfs.values():
            for a in surfs:
                for b in surfs:
                    if a != b:
                        same_stack_pairs.add((a, b))

        # Build fault directed pairs
        if (stacks_structure is not None
                and stacks_structure.faults_relations is not None):
            faults_relations = stacks_structure.faults_relations
            n_stacks = stacks_structure.n_stacks

            for fs in range(n_stacks):
                for ds in range(n_stacks):
                    if faults_relations[fs, ds]:
                        for fi in _stack_to_surfs[fs]:
                            for di in _stack_to_surfs[ds]:
                                fault_directed_pairs.add((fi, di))

    # Collect all surface indices involved in fault relations (either direction)
    fault_involved_pairs: set = set()  # both (i,j) and (j,i)
    for fi, di in fault_directed_pairs:
        fault_involved_pairs.add((fi, di))
        fault_involved_pairs.add((di, fi))

    # Also exclude fault–fault pairs (both surfaces belong to fault stacks)
    if (surface_to_stack is not None
            and stacks_structure is not None
            and stacks_structure.faults_relations is not None):
        faults_relations = stacks_structure.faults_relations
        n_stacks = stacks_structure.n_stacks
        is_fault_stack = set()
        for fs in range(n_stacks):
            for ds in range(n_stacks):
                if faults_relations[fs, ds]:
                    is_fault_stack.add(fs)
                    break
        for i in range(n):
            for j in range(i + 1, n):
                si = surface_to_stack[i]
                sj = surface_to_stack[j]
                if si in is_fault_stack and sj in is_fault_stack:
                    fault_involved_pairs.add((i, j))
                    fault_involved_pairs.add((j, i))

    # --- 1. Averaging for erosion/onlap pairs only -------------------------
    # Only average vertices between surfaces that are in different stacks
    # and NOT linked by a fault relation.  Same-stack and fault pairs are
    # excluded entirely.
    #
    # Vectorized approach: process each eligible pair (i, j) and average
    # their shared-voxel vertices using numpy intersections.

    # Build the set of averaging-eligible surface pairs
    avg_pairs: list = []  # list of (i, j) with i < j
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) in same_stack_pairs:
                continue
            if (i, j) in fault_involved_pairs:
                continue
            avg_pairs.append((i, j))

    # For each eligible pair, average vertices at shared voxels
    for i, j in avg_pairs:
        common, idx_i, idx_j = BackendTensor.t.intersect1d(
            codes[i], codes[j],
            assume_unique=True, return_indices=True
        )
        if common.size == 0:
            continue
        avg = (all_meshes[i].vertices[idx_i] + all_meshes[j].vertices[idx_j]) / 2.0
        all_meshes[i].vertices[idx_i] = avg
        all_meshes[j].vertices[idx_j] = avg

    # --- 2. Fault→layer: copy fault vertex to layer -----------------------
    for fi, di in fault_directed_pairs:
        common, idx_fi, idx_di = BackendTensor.t.intersect1d(
            codes[fi], codes[di],
            assume_unique=True, return_indices=True
        )
        if common.size == 0:
            continue
        # Layer takes the fault's vertex directly (fault keeps its own)
        all_meshes[di].vertices[idx_di] = all_meshes[fi].vertices[idx_fi]


def remove_fault_overlap_triangles(
        all_meshes: List[DualContouringMesh],
        left_right_per_mesh: List[np.ndarray],
        base_number: tuple[int, int, int],
        surface_to_stack: List[int],
        stacks_structure: 'StacksStructure',
        max_workers: int = None
) -> None:
    from ...modules.dual_contouring._find_vertex_overlap import _generate_voxel_codes

    faults_relations = stacks_structure.faults_relations
    if faults_relations is None or len(all_meshes) < 2:
        return

    codes = _generate_voxel_codes(left_right_per_mesh, base_number)
    n_stacks = stacks_structure.n_stacks

    # 1. Pre-calculate which surface indices belong to which stack
    stack_to_surfaces = defaultdict(list)
    for si, stack_idx in enumerate(surface_to_stack):
        stack_to_surfaces[stack_idx].append(si)

    # 2. Build tasks Grouped by DESTINATION surface
    dest_to_faults = defaultdict(list)
    for fault_stack in range(n_stacks):
        for dest_stack in range(n_stacks):
            if not faults_relations[fault_stack, dest_stack]:
                continue

            # Map every fault surface to the destination surface it affects
            for fi in stack_to_surfaces[fault_stack]:
                for di in stack_to_surfaces[dest_stack]:
                    dest_to_faults[di].append(fi)

    if not dest_to_faults:
        return

    # 3. Process each destination mesh entirely in its own thread
    def _process_destination_mesh(di: int, fault_indices: List[int]):
        # Combine all fault codes targeting this mesh into one array
        combined_fault_codes = np.concatenate([codes[fi] for fi in fault_indices])

        # Fast unique to reduce intersection workload
        unique_fault_codes = np.unique(combined_fault_codes)

        # assume_unique=True is safe here if codes[di] has no internal duplicates
        common = np.intersect1d(codes[di], unique_fault_codes, assume_unique=True)

        if common.size == 0:
            return

        mask_d = np.isin(codes[di], common)
        dest_voxel_indices = np.where(mask_d)[0]

        # Execute the heavy mesh mutation INSIDE the thread. 
        # Since each thread works on a different 'di', this is entirely thread-safe!
        remove_triangles_in_voxels(
            mesh=all_meshes[di],
            voxel_indices=dest_voxel_indices,
            mode='all'
        )

    # 4. Launch the threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
                executor.submit(_process_destination_mesh, di, faults)
                for di, faults in dest_to_faults.items()
        ]
        # Wait for all mesh mutations to finish
        concurrent.futures.wait(futures)
