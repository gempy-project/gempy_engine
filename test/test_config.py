import gempy_engine.config
import gempy_engine
import gempy_engine.systems.kernel.kernel


def test_gempy_tensor():
    tfnp = gempy_engine.systems.kernel.kernel.tfnp
    print(tfnp.__version__)
    gempy_engine.config.use_tf = True
    print(tfnp.__version__)

    gempy_engine.config.gempy_tensor.use_tf = True
    print(gempy_engine.config.gempy_tensor.tfnp is tfnp )
    print(tfnp.__version__)

