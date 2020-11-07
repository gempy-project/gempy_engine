import os
os.environ["JAX_PLATFORM_NAME"] = 'cpu' # gpu

from jax import grad
import jax.numpy as jnp
from jax import device_put
import jax


def test_jax_runs():

    def tanh(x):  # Define a function
        y = jnp.exp(-2.0 * x)
        return (1.0 - y) / (1.0 + y)

    grad_tanh = grad(tanh)  # Obtain its gradient function
    print(grad_tanh(1.0))
    print(device_put(1, jax.devices()[-1]).device_buffer.device())