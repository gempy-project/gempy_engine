# # import gempy_engine.config_
# # import gempy_engine
# # import gempy_engine.systems.kernel.kernel
#
#
# def test_gempy_tensor():
#     tfnp = gempy_engine.systems.kernel.kernel.tfnp
#     print(tfnp.__version__)
#     gempy_engine.config_.use_tf = True
#     print(tfnp.__version__)
#
#     gempy_engine.config_.gempy_tensor.use_tf = True
#     print(gempy_engine.config_.gempy_tensor.tfnp is tfnp)
#     print(tfnp.__version__)


def test_optional_dependencies():
    import gempy_engine.config

    print(gempy_engine.config.is_numpy_installed)
    print(gempy_engine.config.is_tensorflow_installed)