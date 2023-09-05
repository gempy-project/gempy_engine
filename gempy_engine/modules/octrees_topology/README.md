### The octree to regular grid equivalence:

- n_levels = 0 -> 32
- n_levels = 1 -> 128
- n_levels = 2 -> 512
- n_levels = 3 -> 2048
- n_levels = 4 -> 8192
- n_levels = 5 -> 32.768
- n_levels = 6 -> 131072
- n_levels = 7 -> 524288 === PURE Numpy in the GB range ===
- n_levels = 8 -> 2097152  === Pykeops without a sweat ===
- n_levels = 9 -> 8388608   === HELL ZONE === Solutions matrices are too large
- n_levels = 10 -> 33554432