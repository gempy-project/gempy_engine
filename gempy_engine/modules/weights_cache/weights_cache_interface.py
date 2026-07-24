import hashlib
import os
import pickle
import tempfile
import threading
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
    _lock = threading.RLock()

    @staticmethod
    def initialize_cache_dir(disk_cache_dir=None):
        with WeightCache._lock:
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
        with WeightCache._lock:
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
    def store_weights(file_name, hash, weights, write_to_disk: bool = True):
        cache_entry = {
            "hash": hash,
            "weights": weights,
        }
        with WeightCache._lock:
            WeightCache.memory_cache[file_name] = cache_entry
            if not write_to_disk:
                return

            final_path = WeightCache._disk_cache_path(file_name)
            file_descriptor, temporary_path = tempfile.mkstemp(
                prefix=f"{file_name}.",
                suffix=".tmp",
                dir=WeightCache.disk_cache_dir,
            )
            try:
                with os.fdopen(file_descriptor, "wb") as cache_file:
                    pickle.dump(cache_entry, cache_file)
                os.replace(temporary_path, final_path)
            finally:
                if os.path.exists(temporary_path):
                    os.remove(temporary_path)

    @staticmethod
    def load_weights(key, look_in_disk: bool) -> Optional[dict]:
        with WeightCache._lock:
            if key in WeightCache.memory_cache:
                return WeightCache.memory_cache[key]

            if not look_in_disk:
                return None
            disk_path = WeightCache._disk_cache_path(key)
            if os.path.exists(disk_path):
                with open(disk_path, "rb") as cache_file:
                    try:
                        weights = pickle.load(cache_file)
                    except (ModuleNotFoundError, EOFError, pickle.UnpicklingError):
                        return None
                WeightCache.memory_cache[key] = weights
                return weights

        return None


WeightCache.initialize_cache_dir()  # Initialize with default or provide custom path
