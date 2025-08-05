import hashlib
import os
import pickle
import tempfile
from typing import Optional


def generate_cache_key(name, parameters):
    # Example of creating a composite key
    param_hash = hashlib.md5(repr(parameters).encode()).hexdigest()
    return f"group_{name}_params_{param_hash}"


class WeightCache:
    memory_cache = {}
    disk_cache_dir = None

    max_size_mb = 50
    reduce_to_mb = 25

    @staticmethod
    def initialize_cache_dir(disk_cache_dir=None):
        if disk_cache_dir is None:
            # Use a subdirectory in the system's temp directory
            temp_dir = tempfile.gettempdir()
            WeightCache.disk_cache_dir = os.path.join(temp_dir, "gempy_cache")
        else:
            WeightCache.disk_cache_dir = disk_cache_dir

        os.makedirs(WeightCache.disk_cache_dir, exist_ok=True)
        WeightCache._check_and_cleanup_cache()
    
    @staticmethod
    def clear_cache():
        WeightCache.memory_cache = {}
        WeightCache._check_and_cleanup_cache()
    
    @staticmethod
    def _check_and_cleanup_cache():
        total_size = 0
        file_list = []

        for filename in os.listdir(WeightCache.disk_cache_dir):
            file_path = os.path.join(WeightCache.disk_cache_dir, filename)
            if os.path.isfile(file_path):
                file_size = os.path.getsize(file_path)
                total_size += file_size
                file_list.append((file_path, file_size, os.path.getmtime(file_path)))

        if total_size > WeightCache.max_size_mb * 1024 * 1024:
            # Sort files by modified time (oldest first)
            file_list.sort(key=lambda x: x[2])

            # Remove files until size is below the reduce_to_mb threshold
            size_to_reduce = total_size - WeightCache.reduce_to_mb * 1024 * 1024
            for file_path, file_size, _ in file_list:
                if size_to_reduce > 0:
                    os.remove(file_path)
                    size_to_reduce -= file_size
                else:
                    break
    @staticmethod
    def _disk_cache_path(key):
        return os.path.join(WeightCache.disk_cache_dir, f"{key}.pkl")

    @staticmethod
    def store_weights(file_name, hash, weights):
        
        # Store in memory
        WeightCache.memory_cache[file_name] = {
            "hash": hash,
            "weights": weights
        }

        # Optionally store on disk as well
        with open(WeightCache._disk_cache_path(file_name), 'wb') as f:
            pickle.dump(
                {
                    "hash": hash,
                    "weights": weights
                },
                f)

    @staticmethod
    def load_weights(key, look_in_disk: bool) -> Optional[dict]:
        # Try to load from memory
        if key in WeightCache.memory_cache:
            return WeightCache.memory_cache[key]

        # Load from disk if not in memory
        if not look_in_disk:
            return None
        disk_path = WeightCache._disk_cache_path(key)
        if os.path.exists(disk_path):
            with open(disk_path, 'rb') as f:
                try:
                    weights = pickle.load(f)
                except ModuleNotFoundError:
                    # Handle case where the module has been renamed
                    # and the pickled object cannot be loaded
                    return None
                # Optionally cache in memory again
                WeightCache.memory_cache[key] = weights
                return weights

        return None


WeightCache.initialize_cache_dir()  # Initialize with default or provide custom path
