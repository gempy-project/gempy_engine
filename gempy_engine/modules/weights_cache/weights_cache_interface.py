import os
import pickle

import os
import hashlib
import tempfile


def generate_cache_key(name, parameters):
    # Example of creating a composite key
    param_hash = hashlib.md5(repr(parameters).encode()).hexdigest()
    return f"group_{name}_params_{param_hash}"


def get_default_cache_dir():
    # Use a subdirectory in the system's temp directory
    temp_dir = tempfile.gettempdir()
    cache_dir = os.path.join(temp_dir, "gempy_cache")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


class WeightCache:
    """
    # Example usage
    iteration = 5
    parameters = {'param1': 'value1', 'param2': 'value2'}
    cache_key = generate_cache_key(iteration, parameters)
    cache_dir = get_default_cache_dir()
    
    print("Cache Key:", cache_key)
    print("Cache Directory:", cache_dir)

    """

    def __init__(self, disk_cache_dir):
        self.memory_cache = {}
        self.disk_cache_dir = disk_cache_dir
        os.makedirs(disk_cache_dir, exist_ok=True)

    def _disk_cache_path(self, key):
        return os.path.join(self.disk_cache_dir, f"{key}.pkl")

    def store_weights(self, key, weights):
        # Store in memory
        self.memory_cache[key] = weights

        # Optionally store on disk as well
        with open(self._disk_cache_path(key), 'wb') as f:
            pickle.dump(weights, f)

    def load_weights(self, key):
        # Try to load from memory
        if key in self.memory_cache:
            return self.memory_cache[key]

        # Load from disk if not in memory
        disk_path = self._disk_cache_path(key)
        if os.path.exists(disk_path):
            with open(disk_path, 'rb') as f:
                weights = pickle.load(f)
                # Optionally cache in memory again
                self.memory_cache[key] = weights
                return weights

        # Handle case where weights are not cached
        raise KeyError(f"Weights for key {key} not found in cache.")



