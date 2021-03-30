# gempy_engine


## GPU Install in Linux


- Check the instalation guide of tensorflow. It is very picky on the enviroment
- Install nvidia drivers before trying to install cuda
    - the cuda installer has also the drivers but apparently is more difficult to set up
- Download the .run file instead the deb file
- cuDNN is always required an it is a pain
  + this has to be installed with dev because it is IMPORTANT to install:
    - Runtime library
    - developer library
  
### pykeops
  - It needs cmake in the enviroment
  - Make sure this is in bashrc:

>>>  export PATH="/usr/local/cuda-11.0/bin:$PATH"
>>>  export LD_LIBRARY_PATH="/usr/local/cuda-8.0/lib64:$LD_LIBRARY_PATH"