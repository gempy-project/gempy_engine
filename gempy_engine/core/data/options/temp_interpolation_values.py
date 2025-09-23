from dataclasses import dataclass


@dataclass
class TempInterpolationValues:
    current_octree_level: int = 0  # * Make this a read only property 
    start_computation_ts: int = -1
