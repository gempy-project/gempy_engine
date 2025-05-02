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


### Adding environment variables:

>  export PATH="/usr/local/cuda-11.2/bin:$PATH"
> 
>  export LD_LIBRARY_PATH="/usr/local/cuda-8.0/lib64:$LD_LIBRARY_PATH"

for WSL:
>  export PATH=/usr/local/cuda/bin:$PATH
> 
>  export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH

> **Important**: Make sure that your Python-dev version matches your environment python version.

Adding the path to bash is not enough for Pycharm. It has to be added to the enviroment variables. In ubuntu is on the file `/etc/environment`. Edit it with the following command `sudo -H gedit /etc/environment`.

You can check that the variables are properly set in Pycharm looking in the Run Config

