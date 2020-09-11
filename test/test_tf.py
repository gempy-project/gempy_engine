import pytest
import tensorflow as tf
import numpy as np


def test_gpu_available():
    tf.test.is_gpu_available()


def test_tensor_vs_variable():
    a = tf.Variable(tf.tile(np.array([1]), [3]))
    a.assign(np.ones(3))
    print(a)
    with pytest.raises(AttributeError):
        a = tf.tile(np.array([1]), [3])
        print('Coming from op: ', a)
        a.assign(np.ones(3))


def test_tf_conts():
    print(tf.constant([5, 5, 5]))