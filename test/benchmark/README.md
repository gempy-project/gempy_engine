To profile memory using the `memory-profiler` package, run the following command:
```
python -m memory_profiler benchmark.py
mprof run <executable>
mprof plot
``` 

### Compare previous runs

`pytest-benchmark compare 0001 0002`



### List of optimizations

- [x]  Caching weights
- [ ]  Use new fancy triangulation
- [ ]  fancy triangulation in gpu
  (after profiling)
- [ ]  dtype
- [ ] Gradients calculations (we only need it on the edges). More info on Notability->GemPy Engine Page 60


### Scalene Profiling
The only thing I have been able to run consistenly is with the arg profile only

`scalene --profile-all --profile-only model,interp,octree,scalar_field,solver_interface,kernel,vectors profile_runner.py`  
