#! python3

import unittest
import numpy as np

from variable import Variable
from function import *

class AddTest(unittest.TestCase) :
  def test_forward(self) :
    inputs = [Variable(np.array(1.0)), Variable(np.array(1.0))]
    outputs = add(inputs)
    expected = np.array(2.0)
    self.assertEqual(outputs[0].data, expected)

unittest.main()