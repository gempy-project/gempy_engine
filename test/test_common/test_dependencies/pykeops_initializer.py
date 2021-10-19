import pykeops

pykeops.verbose = True
    #pykeops.set_bin_folder("/home/miguel/.s")
pykeops.clean_pykeops()  # just in case old build files are still present
#pykeops.config.build_type = 'Debug'
pykeops.test_numpy_bindings()

