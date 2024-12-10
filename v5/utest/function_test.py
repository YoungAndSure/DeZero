#! python3

if '__file__' in globals() :
  import os, sys
  sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import unittest
import numpy as np
import math

from dezero.core import *
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
  
  def test_tan(self) :
    x = Variable(np.array(np.pi))
    y = tanh(x)
    self.assertTrue(np.allclose(y.data, 0.996272))
    y.backward()
    self.assertTrue(np.allclose(x.grad.data, 0.00744195))
  
  def test_tensor_add(self) :
    x = Variable(np.array([1.0, 2.0, 3.0]))
    y = Variable(np.array([3.0, 2.0, 1.0]))
    z = x + y
    self.assertTrue(np.array_equal(z.data, [4.0, 4.0, 4.0]))
    z.backward()
    self.assertTrue(np.array_equal(x.grad.data, [1.0, 1.0, 1.0]))
  
  def test_reshape(self) :
    t = np.array([[1.0, 2.0, 3.0],[4.0, 5.0, 6.0]])
    x = Variable(t)
    y = reshape(x, (6, ))
    self.assertTrue(np.array_equal(y.data, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
    y.backward()
    self.assertTrue(np.array_equal(x.grad.data, [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]))
    z = reshape(t, (2, 3))
    self.assertTrue(np.array_equal(z.data, t))
    z.backward()
  
  def test_variable_reshape(self) :
    x = Variable(np.array([[1.0, 2.0, 3.0],[4.0, 5.0, 6.0]]))
    y = x.reshape((6, ))
    self.assertTrue(np.array_equal(y.data, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
    y.backward()
    self.assertTrue(np.array_equal(x.grad.data, [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]))

    x = Variable(np.array([[1.0, 2.0, 3.0],[4.0, 5.0, 6.0]]))
    y = x.reshape(6, )
    self.assertTrue(np.array_equal(y.data, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
    y.backward()
    self.assertTrue(np.array_equal(x.grad.data, [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]))

  def test_transpose(self) :
    x = Variable(np.array([[1.0, 2.0, 3.0],[4.0, 5.0, 6.0]]))
    y = transpose(x)
    self.assertTrue(np.array_equal(y.data, [[1.0,4.0], [2.0, 5.0], [3.0, 6.0]]))
    y.backward()
    self.assertTrue(np.array_equal(x.grad.data, [[1.0, 1.0, 1.0],[1.0, 1.0, 1.0]]))

    x.cleargrad()
    y = x.transpose()
    self.assertTrue(np.array_equal(y.data, [[1.0,4.0], [2.0, 5.0], [3.0, 6.0]]))
    y.backward()
    self.assertTrue(np.array_equal(x.grad.data, [[1.0, 1.0, 1.0],[1.0, 1.0, 1.0]]))

    x.cleargrad()
    y = x.T
    self.assertTrue(np.array_equal(y.data, [[1.0,4.0], [2.0, 5.0], [3.0, 6.0]]))
    y.backward()
    self.assertTrue(np.array_equal(x.grad.data, [[1.0, 1.0, 1.0],[1.0, 1.0, 1.0]]))

    x = Variable(np.array([1.0, 2.0, 3.0]))
    y = x.T
    self.assertTrue(np.array_equal(y.data, [[1.0], [2.0], [3.0]]))
    y.backward()
    self.assertTrue(np.array_equal(x.grad.data, [1.0, 1.0, 1.0]))

  def test_sum(self) :
    x = Variable(np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
    y = sum(x)
    self.assertTrue(np.array_equal(y.data, 21.0))
    y.backward()
    self.assertTrue(np.array_equal(x.grad.data, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))

    x = Variable(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    y = sum(x)
    self.assertTrue(np.array_equal(y.data, 21.0))
    y.backward()
    self.assertTrue(np.array_equal(x.grad.data, [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]))

    x = Variable(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    y = sum(x, keepdims=True)
    self.assertTrue(np.array_equal(y.data, [[21.0]]))
    y.backward()
    self.assertTrue(np.array_equal(x.grad.data, [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]))

  def test_broadcast2_sum2(self) :
    x = Variable(np.array([1.0, 2.0, 3.0]))
    y = broadcast_to(x, (2, 3))
    self.assertTrue(np.array_equal(y.data, [[1.0,2.0,3.0],[1.0,2.0,3.0]]))
    y.backward()
    self.assertTrue(np.array_equal(x.grad.data, [2.0, 2.0, 2.0]))

    x.cleargrad()
    z = sum_to(y, (1,3))
    self.assertTrue(np.array_equal(z.data, [[2.0, 4.0, 6.0]]))
    z.backward(retain_grad=True)
    self.assertTrue(np.array_equal(y.grad.data, [[1.0,1.0,1.0],[1.0,1.0,1.0]]))
  
  def test_support_broadcast2_add(self) :
    x = Variable(np.array([1.0, 2.0, 3.0]))
    y = Variable(np.array([1.0]))
    z = x + y
    self.assertTrue(np.array_equal(z.data, [2.0, 3.0, 4.0]))
    z.backward()
    self.assertTrue(np.array_equal(x.grad.data, [1.0, 1.0, 1.0]))
    self.assertTrue(np.array_equal(y.grad.data, [3.0]))
  
  def test_matmul(self) :
    x = Variable(np.array([[1.0, 2.0, 3.0]]))
    W = Variable(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
    y = matmul(x, W)
    self.assertTrue(np.array_equal(y.data, np.array([[22, 28]])))
    y.backward()
    self.assertTrue(np.array_equal(x.grad.data, np.array([[3.0, 7.0, 11.0]])))

  def test_mean_square(self) :
    x0 = Variable(np.array([4,5,6]))
    x1 = Variable(np.array([7,8,9]))
    y = mean_square(x0, x1)
    self.assertTrue(np.array_equal(y.data, np.array([3.0, 3.0, 3.0])))
    y.backward()
    self.assertTrue(np.array_equal(x0.grad.data, [-2.0, -2.0, -2.0]))

unittest.main()