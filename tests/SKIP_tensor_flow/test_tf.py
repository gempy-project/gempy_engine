import pytest
tf = pytest.importorskip("tensorflow")
import numpy as np


@pytest.mark.skip('Only trigger manually when there is something wrong with'
                  'pykeops compilation', )
def test_tensor_vs_variable():
    a = tf.Variable(tf.tile(np.array([1]), [3]))
    a.assign(np.ones(3))
    print(a)
    with pytest.raises(AttributeError):
        a = tf.tile(np.array([1]), [3])
        print('Coming from op: ', a)
        a.assign(np.ones(3))

@pytest.mark.skip('Only trigger manually when there is something wrong with'
                  'pykeops compilation', )
def test_tf_conts():
    print(tf.constant([5, 5, 5]))
    print(tf.config.experimental.list_physical_devices())

