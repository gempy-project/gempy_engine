from gempy_engine.config import BackendTensor, AvailableBackends


def test_xla_surface_points_preprocessing(simple_model_2):
    surface_points = simple_model_2[0]
    tensors_structure = simple_model_2[3]
    BackendTensor.change_backend(AvailableBackends.tensorflow, use_gpu=False)
    from gempy_engine.modules.kernel_constructor._input_preparation import surface_points_preprocess
    import tensorflow as tf
    @tf.function
    def tf_f(surface_points, tensors_structure):
        static =  surface_points_preprocess(surface_points, tensors_structure.number_of_points_per_surface)
        return static.ref_surface_points

    s = tf_f(surface_points, tensors_structure)
    print(s)
