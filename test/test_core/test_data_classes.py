from gempy_engine.core.data.data_shape import TensorsStructure
import numpy as np


def test_changing_dtype_in_post_init():

    _ = np.ones(3)
    tensor_structure = TensorsStructure(np.array([3, 2]), _, _, _, _)
    print(tensor_structure)