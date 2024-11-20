#! python3

import unittest
import numpy as np

from variable import Variable
from function import *

class AddTest(unittest.TestCase) :
  def test_forward(self) :
    x0 = Variable(np.array(1.0))
    x1 = Variable(np.array(1.0))
    outputs = add(x0, x1)
    expected = np.array(2.0)
    self.assertEqual(outputs[0].data, expected)

unittest.main()