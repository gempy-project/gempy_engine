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
        # if self._number_octree_levels_surface >= self.number_octree_levels -1:
        #     return self.number_octree_levels 
        # else:
        #     return self._number_octree_levels_surface
        # ? Should we make this just a property
        return self._number_octree_levels_surface

    @number_octree_levels_surface.setter
    def number_octree_levels_surface(self, value):
        # Check value is between 1 and number_octree_levels
        if not 1 <= value:
            raise ValueError("number_octree_levels_surface must be at least 1")
        self._number_octree_levels_surface = value
