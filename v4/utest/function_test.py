#! python3

if '__file__' in globals() :
  import os, sys
  sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import unittest
import numpy as np
import math

from dezero.variable import *
from dezero.function import *
from dezero.config import *
from dezero.user_defined_func import *
from dezero.util import _dot_var, _dot_func, get_dot_graph, plot_dot_graph

class AddTest(unittest.TestCase) :
  def test_backward(self) :
    x0 = Variable(np.array(1.0))
    x1 = Variable(np.array(1.0))
    outputs = add(x0, x1)
    outputs.backward()
    expected = np.array(1.0)
    self.assertEqual(x0.grad.data, expected)
    self.assertEqual(x1.grad.data, expected)
  
  def test_second_backward(self) :
    x = Variable(np.array(2.0))
    y = x ** 4 - 2 * x ** 2
    y.backward(create_graph=True)
    self.assertEqual(x.grad.data, 24.0)

    gx = x.grad
    x.cleargrad()
    gx.backward()
    # 注意，二阶导数不在y.grad，也不在gx.grad，而是传回了x.grad
    # 二次反向传播先是传回了y.grad，然后又继续往回传到了x.grad
    self.assertEqual(x.grad.data, 44.0)

  def test_cos(self) :
    x = Variable(np.array(np.pi / 4))
    y = cos(x)
    self.assertTrue(np.allclose(y.data, 0.70710678))
    y.backward()
    self.assertTrue(np.allclose(x.grad.data, -0.70710678))

unittest.main()