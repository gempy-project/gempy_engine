import enum

from dataclasses import dataclass


class MeshExtractionMaskingOptions(enum.Enum):
    NOTHING = enum.auto()  # * This is only for testing
    DISJOINT = enum.auto()
    INTERSECT = enum.auto()
    RAW = enum.auto()


@dataclass
class EvaluationOptions:
    _number_octree_levels: int = 1
    _number_octree_levels_surface: int = 4
    octree_curvature_threshold: float = -1.  #: Threshold to do octree refinement due to curvature to deal with angular geometries. This curvature assumes that 1 is the maximum curvature of any voxel
    octree_error_threshold: float = 1.  #: Number of standard deviations to consider a voxel as candidate to refine
    octree_min_level: int = 2
    
    mesh_extraction: bool = True
    mesh_extraction_masking_options: MeshExtractionMaskingOptions = MeshExtractionMaskingOptions.INTERSECT
    mesh_extraction_fancy: bool = True

    evaluation_chunk_size: int = 500_000

    compute_scalar_gradient: bool = False
    
    verbose: bool = False

    @property
    def number_octree_levels(self):
        # Return whatever is bigger between the number of octree levels and the number of octree levels for surfaces
        if self.mesh_extraction and self._number_octree_levels < 2 and self._number_octree_levels_surface >= self._number_octree_levels:
            return self._number_octree_levels_surface
        
        return self._number_octree_levels

    @number_octree_levels.setter
    def number_octree_levels(self, value):
        # Check value is at least 1
        if not 1 <= value:
            raise ValueError("number_octree_levels must be at least 1")
        self._number_octree_levels = value

    @property
    def number_octree_levels_surface(self):
        # Raise error if the number of octree levels for the surface is 0
        if self._number_octree_levels_surface <= 1:
            raise ValueError("The number of octree levels for the surface must be greater than 1.")

        if self._number_octree_levels_surface >= self.number_octree_levels:
            return self.number_octree_levels 
        return self._number_octree_levels_surface

    @number_octree_levels_surface.setter
    def number_octree_levels_surface(self, value):
        # Check value is between 1 and number_octree_levels
        if not 1 <= value:
            raise ValueError("number_octree_levels_surface must be at least 1")
        self._number_octree_levels_surface = value
