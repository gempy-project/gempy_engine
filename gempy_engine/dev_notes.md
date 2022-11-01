TODO:

- [ ] Refactor code: Refactor until exporting scalarfield and gradient
    - [X] weights
    - [X] scalar field export
    - [X] gradient export
        - **Interpolate model with gempy and get scalar field** different u_terms!
        - compute gradients with numpy
    - [X] remove unsued code
    - [X] refactor TensorBackend      

- [ ] Add drift to the export

- [ ] Keep the range cont by increasing r

- [ ] Gradient to scalar field
    - Mixing xla-eager
    - putting keops in the middle

- [ ] Test in the covariance module if we can mix eager and xla

- [ ] Add benchmark test
    - Small medium and big model
    - xla, pykeops
    
- [ ] Finite faults: the ellipse can be precomputed probably as a getter of its data class!

- [ ] Data
    - [ ] `__post_init__` should go to utils
    - [ ] Add hash function to all data classes




Notes
=====

- Do not try to reinvent the wheel follow tensorflow intalation guide for gpu
    - Tensorflow instalation guide install a gcc and g++ compiler and set them default
    - Use the following to change the default compiler. 
        - `sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 50`
        - `sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 50
          `- Intalling pykeops is a nightmare:
            - Bash looks like this
          ```      
          export PATH="/usr/local/cuda-11.2/bin:$PATH"
          export LD_LIBRARY_PATH="/usr/local/cuda-11.2/lib64:$LD_LIBRARY_PATH"
          export CPLUS_INCLUDE_PATH="/usr/local/cuda/include"
          
          export CC=/usr/bin/gcc-9
          export CXX=/usr/bin/g++-9
          ```
            - **sudo apt-get install python3.8-dev** This was fundamental for a pybind11 error
            - In the end I am using pip install in venv

- Remember to type the code

- **XLA** Can be applied just in parts of the subtree using jit_scope
    - ! (2022) The performance test I did show that is not worth it
    - To use XLA the tensorflow function must be defined!
    - [ ] Test with the covartiance