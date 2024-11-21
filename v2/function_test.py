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

  def test_backward(self) :
    x0 = Variable(np.array(1.0))
    x1 = Variable(np.array(1.0))
    outputs = add(x0, x1)
    outputs[0].backward()
    expected = np.array(1.0)
    self.assertEqual(x0.grad, expected)
    self.assertEqual(x1.grad, expected)
  
  def test_square(self) :
    x = Variable(np.array(3.0))
    y = Variable(np.array(3.0))
    z = add(square(x), square(y))
    self.assertEqual(z[0].data, np.array(18.0))
    z[0].backward()

    self.assertEqual(x.grad, np.array(6.0))
    self.assertEqual(y.grad, np.array(6.0))

unittest.main()