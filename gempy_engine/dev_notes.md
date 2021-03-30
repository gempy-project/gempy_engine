TODO:

- [ ] Refactor code

- [ ] Data `__post_init__` should go to utils

- [ ] Test in the covariance module if we can mix eager and function

- [ ] Keep the range cont by increasing r

- [ ]  NOTE: Interesting enough this Gx and Gy are swapped between numpy and pykeops TODO: Check that it does not have
  effect in the scalar field

Notes
=====

- Do not try to reinvent the wheel follow tensorflow intalation guide for gpu
    - Tensorflow instalation guide install a gcc and g++ compiler and set them default
    - Use the following to change the default compiler. **Pykeops** needs gcc 8 (April 2021)
        - `sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 50`
        - `sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 50
          `- Intalling pykeops is a nightmare:
            - Bash looks like this
          ```      
          export PATH="/usr/local/cuda-11.0/bin:$PATH"
          export LD_LIBRARY_PATH="/usr/local/cuda-11.0/lib64:$LD_LIBRARY_PATH"
          export CPLUS_INCLUDE_PATH="/usr/local/cuda/include"
          
          export CC=/usr/bin/gcc-9
          export CXX=/usr/bin/g++-9
          ```
            - **sudo apt-get install python3.8-dev** This was fundamental for a pybind11 error
            - In the end I am using pip install in venv

- Remember to type the code

- **XLA** Can be applied just in parts of the subtree using jit_scope
    - To use XLA the tensorflow function must be defined!
    - [ ] Test with the covartiance