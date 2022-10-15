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
- [ ] Opt2: Use new fancy triangulation
- [ ] Opt3: fancy triangulation in gpu
  (after profiling)
- [ ] Opt4 dtype



### Scalene Profiling
The only thing I have been able to run consistenly is with the arg profile only

`scalene --profile-all --profile-only model,interp,octree,scalar_field,solver_interface,kernel,vectors profile_runner.py`  


### Notes:

- Numpy:
  - LowInput Opt2 Octtree 5: 56% interpolate 44% dual contouring
  - LowInput Opt2 Octtree 4: 
    - func compute_model: 57% interpolate 42% dual contouring
    - func dual_contouring_multi_scalar: 96.9% interpolate_all_fields_no_octree 3.1% compute_dual_contouring
    - func compute_dual_contouring: 96.3 triangulate 3.7% vertices