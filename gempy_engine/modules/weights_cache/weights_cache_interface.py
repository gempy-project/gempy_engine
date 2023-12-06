import hashlib
import os
import pickle
import tempfile


def generate_cache_key(name, parameters):
    # Example of creating a composite key
    param_hash = hashlib.md5(repr(parameters).encode()).hexdigest()
    return f"group_{name}_params_{param_hash}"


class WeightCache:
    memory_cache = {}
    disk_cache_dir = None

    @staticmethod
    def initialize_cache_dir(disk_cache_dir=None):
        if disk_cache_dir is None:
            # Use a subdirectory in the system's temp directory
            temp_dir = tempfile.gettempdir()
            WeightCache.disk_cache_dir = os.path.join(temp_dir, "gempy_cache")
        else:
            WeightCache.disk_cache_dir = disk_cache_dir

        os.makedirs(WeightCache.disk_cache_dir, exist_ok=True)

    @staticmethod
    def _disk_cache_path(key):
        return os.path.join(WeightCache.disk_cache_dir, f"{key}.pkl")

    @staticmethod
    def store_weights(key, weights):
        # Store in memory
        WeightCache.memory_cache[key] = weights

        # Optionally store on disk as well
        with open(WeightCache._disk_cache_path(key), 'wb') as f:
            pickle.dump(weights, f)

    @staticmethod
    def load_weights(key):
        # Try to load from memory
        if key in WeightCache.memory_cache:
            return WeightCache.memory_cache[key]

        # Load from disk if not in memory
        disk_path = WeightCache._disk_cache_path(key)
        if os.path.exists(disk_path):
            with open(disk_path, 'rb') as f:
                weights = pickle.load(f)
                # Optionally cache in memory again
                WeightCache.memory_cache[key] = weights
                return weights

        return None


WeightCache.initialize_cache_dir()  # Initialize with default or provide custom path
