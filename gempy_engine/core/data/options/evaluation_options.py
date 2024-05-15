import enum

from dataclasses import dataclass

from ..raw_arrays_solution import RawArraysSolution


class MeshExtractionMaskingOptions(enum.Enum):
    NOTHING = enum.auto()  # * This is only for testing
    DISJOINT = enum.auto()
    INTERSECT = enum.auto()
    RAW = enum.auto()


@dataclass
class EvaluationOptions:
    _number_octree_levels: int = 1
    _number_octree_levels_surface: int = 4
    curvature_threshold: float = 0.8  # * Threshold to do octree refinement due to curvature to deal with angular geometries. This curvature assumes that 1 is the maximum curvature of any voxel
    block_solutions_type: RawArraysSolution.BlockSolutionType = RawArraysSolution.BlockSolutionType.OCTREE

    mesh_extraction: bool = True
    mesh_extraction_masking_options: MeshExtractionMaskingOptions = MeshExtractionMaskingOptions.INTERSECT
    mesh_extraction_fancy: bool = True

    evaluation_chunk_size: int = 50_000

    @property
    def number_octree_levels(self):
        # Return whatever is bigger between the number of octree levels and the number of octree levels for surfaces
        return self._number_octree_levels

    @number_octree_levels.setter
    def number_octree_levels(self, value):
        # Check value is at least 1
        if not 1 <= value:
            raise ValueError("number_octree_levels must be at least 1")
        self._number_octree_levels = value

    @property
    def number_octree_levels_surface(self):
        return self._number_octree_levels_surface

    @number_octree_levels_surface.setter
    def number_octree_levels_surface(self, value):
        # Check value is between 1 and number_octree_levels
        if not 1 <= value:
            raise ValueError("number_octree_levels_surface must be at least 1")
        self._number_octree_levels_surface = value
