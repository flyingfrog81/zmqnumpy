from unittest import TestCase
import zmqnumpy as znp
import numpy as np

class TestSerialization(TestCase):
    def setUp(self):
        self._shape = (10,)
        self._max = 100
        self._random_data = np.random.uniform(0, self._max, self._shape)
        self._msg = znp.array_to_msg(self._random_data)

    def test_array_to_msg_size(self):
        self.assertEqual(len(self._msg), 3)

    def test_array_to_msg_shape(self):
        self.assertEqual(np.fromstring(self._msg[1], np.int32),
                         self._shape)

    def test_array_to_msg_dtype(self):
        self.assertEqual(self._msg[0].decode(),
                         self._random_data.dtype.name)

    def test_array_to_msg(self):
        _data = znp.msg_to_array(self._msg)
        np.testing.assert_array_equal(_data, self._random_data)
