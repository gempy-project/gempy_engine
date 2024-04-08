## Memory Profiling

### Best by far: memray

### Scalene Profiling
The only thing I have been able to run consistenly is with the arg profile only

`scalene --profile-all --profile-only model,interp,octree,scalar_field,solver_interface,kernel,vectors profile_runner.py`

### Memory profiler
To profile memory using the `memory-profiler` package, run the following command:
```
python -m memory_profiler benchmark.py
mprof run <executable>
mprof plot
``` 

# Performance
## Pytest-Benchmark

### Run benchmark

`pytest-benchmark --benchmark-autosave --benchmark-compare `

### Compare previous runs

`pytest-benchmark compare 0001 0002`


### Line profiler is cool 
Use pycharm line profiler. Check out `gempy_profiler_decorator`
- Does not work in wsl2

### List of optimizations
The base octree level is 3. Otherwise, is specified in the name of the file. 

- [x] Opt1: Caching weights
- [x] Opt2: Gradients calculations (we only need it on the edges). More info on Notability->GemPy Engine Page 60. Also reuse distances between gradients
- [x] Opt3: Use new fancy triangulation
- [ ] Opt4: fancy triangulation in gpu
  (after profiling)
- [ ] Opt5 dtype


### Notes:

Analyzing how the computation scales with depth of octrees.

- Numpy:
  - LowInput Opt2 Octtree 5: 56% interpolate 44% dual contouring
  - LowInput Opt2 Octtree 4: 
    - func compute_model: 57% interpolate 42% dual contouring
    - func dual_contouring_multi_scalar: **96.9%** interpolate_all_fields_no_octree 3.1% compute_dual_contouring
    - func compute_dual_contouring: 96.3 vertices  3.7% triangulate
-Pykeops GPU 
  - LowInput Opt2 Octtree 4:
    - func compute_model: **86%** interpolate 13.6% dual contouring
    - func dual_contouring_multi_scalar: **87.5%** interpolate_all_fields_no_octree 12.5% compute_dual_contouring
    - func compute_dual_contouring: 10.4% vertices  89.6% triangulate 
  - LowInput Opt2 Octtree 5:
    - func compute_model: 71.3% interpolate 28.7% dual contouring
    - func dual_contouring_multi_scalar: 28.8% interpolate_all_fields_no_octree **71.2%** compute_dual_contouring
    - func compute_dual_contouring: 1.9% vertices  98.1% triangulate
  - LowInput Opt2 Octtree 6:
    - func compute_model: 20% interpolate 80% dual contouring
    - func dual_contouring_multi_scalar: 2.5% interpolate_all_fields_no_octree **97.5%** compute_dual_contouring
    - func compute_dual_contouring: 0.4% vertices  **98.1**% triangulate
  - LowInput Opt3 Octtree 6:
      - func compute_model: *40%* interpolate 60% dual contouring
      - func dual_contouring_multi_scalar: 7.7% interpolate_all_fields_no_octree **92.2%** compute_dual_contouring
      - func compute_dual_contouring: 1.6% vertices  **98.4**% triangulate
- Pykeops CPU
  - AllInput Opt3 Octtree 3:
      - 1 Iteration with SCIPY_CG takes 4.39 seconds 
      - Convergence rate ( tolerancy of .05 and nugget of 100 ):
        - **float64**: 80X iterations 
        - **float32**: >200 (this was the cap) iterations
    - Convergence rate ( tolerancy of .05 and nugget of 10000 ):
      - **float64**: ?
      - **float32**: 3X  iterations
    - > Running this on the GPU seems to run iteration in 5 ms WTF

Conclusions:
  - **Numpy** after 5 octrees levels becomes very slow on the **kernel side**
  - **Pykeops** The kernel side grows just linearly so after 5 octrees the **triangulation** becomes the bottleneck
  - To test the triangulation optimizations, we need to do it on **octree 5**
  - Moving the triangulation to the GPU is very promising. But it is going to be tricky with memory
   
  - > ! I am suspecting that pykeops CPU was always solving the system of equations on the GPU and therefore the difference was only on the evaluations