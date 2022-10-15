To profile memory using the `memory-profiler` package, run the following command:
```
python -m memory_profiler benchmark.py
mprof run <executable>
mprof plot
``` 


## Pytest-Benchmark

### Compare previous runs

`pytest-benchmark compare 0001 0002`



### List of optimizations
The base octree level is 3. Otherwise, is specified in the name of the file. 

- [x] Opt1: Caching weights
- [x] Opt2: Gradients calculations (we only need it on the edges). More info on Notability->GemPy Engine Page 60. Also reuse distances between gradients
- [ ] Opt3: Use new fancy triangulation
- [ ] Opt4: fancy triangulation in gpu
  (after profiling)
- [ ] Opt5 dtype



### Scalene Profiling
The only thing I have been able to run consistenly is with the arg profile only

`scalene --profile-all --profile-only model,interp,octree,scalar_field,solver_interface,kernel,vectors profile_runner.py`  


### Notes:

Analyzing how the computation scales with depth of octrees.

- Numpy:
  - LowInput Opt2 Octtree 5: 56% interpolate 44% dual contouring
  - LowInput Opt2 Octtree 4: 
    - func compute_model: 57% interpolate 42% dual contouring
    - func dual_contouring_multi_scalar: **96.9%** interpolate_all_fields_no_octree 3.1% compute_dual_contouring
    - func compute_dual_contouring: 96.3 triangulate 3.7% vertices
-Pykeops GPU 
  - LowInput Opt2 Octtree 4:
    - func compute_model: **86%** interpolate 13.6% dual contouring
    - func dual_contouring_multi_scalar: **87.5%** interpolate_all_fields_no_octree 12.5% compute_dual_contouring
    - func compute_dual_contouring: 10.4% triangulate 89.6% vertices 
  - LowInput Opt2 Octtree 5:
    - func compute_model: 71.3% interpolate 28.7% dual contouring
    - func dual_contouring_multi_scalar: 28.8% interpolate_all_fields_no_octree **71.2%** compute_dual_contouring
    - func compute_dual_contouring: 1.9% triangulate 98.1% vertices
  - LowInput Opt2 Octtree 6:
    - func compute_model: 20% interpolate 80% dual contouring
    - func dual_contouring_multi_scalar: 2.5% interpolate_all_fields_no_octree **97.5%** compute_dual_contouring
    - func compute_dual_contouring: 0.4% triangulate **98.1**% vertices

Conclusions:
  - **Numpy** after 5 octrees levels becomes very slow on the **kernel side**
  - **Pykeops** The kernel side grows just linearly so after 5 octrees the **triangulation** becomes the bottleneck
  - To test the triangulation optimizations, we need to do it on **octree 5**