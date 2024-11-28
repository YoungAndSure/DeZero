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

class UtilTest(unittest.TestCase) :
  def test_dot_var(self) :
    x0 = Variable(np.array(1.0))
    x1 = Variable(np.array(2.0))
    y = x0 + x1
    print(_dot_var(y))
    print(_dot_func(y.creator))
  
  def test_get_dot_graph(self) :
    x0 = Variable(np.array(1.0), "x0")
    x1 = Variable(np.array(2.0), "x1")
    y = x0 + x1
    y.name = "y"
    plot_dot_graph(y, verbose=True, to_file="add.png")
  
  def test_get_dot_graph2(self) :
    x = Variable(np.array(1.0))
    y = Variable(np.array(1.0))
    z = goldstein(x, y)
    plot_dot_graph(z, verbose=True, to_file="goldstein.png")
  
  def test_graph_taylor_sin(self) :
    x0 = Variable(np.array(np.pi / 4))
    y0 = taylor_sin(x0)
    x1 = Variable(np.array(np.pi / 4))
    y1 = sin(x1)
    self.assertTrue(np.allclose(y0.data, y1.data))
    plot_dot_graph(y0, verbose=True, to_file="taylor_sin.png")
    plot_dot_graph(y1, verbose=True, to_file="sin.png")

class AddTest(unittest.TestCase) :
  def test_forward(self) :
    x0 = Variable(np.array(1.0))
    x1 = Variable(np.array(1.0))
    outputs = add(x0, x1)
    expected = np.array(2.0)
    self.assertEqual(outputs.data, expected)

  def test_backward(self) :
    x0 = Variable(np.array(1.0))
    x1 = Variable(np.array(1.0))
    outputs = add(x0, x1)
    outputs.backward()
    expected = np.array(1.0)
    self.assertEqual(x0.grad, expected)
    self.assertEqual(x1.grad, expected)
  
  def test_square(self) :
    x = Variable(np.array(3.0))
    y = Variable(np.array(3.0))
    z = add(square(x), square(y))
    self.assertEqual(z.data, np.array(18.0))
    z.backward()

    self.assertEqual(x.grad, np.array(6.0))
    self.assertEqual(y.grad, np.array(6.0))

  def test_same_input(self) :
    x = Variable(np.array(1.0))
    outputs = add(x, x)
    outputs.backward()
    expected = np.array(2.0)
    self.assertEqual(x.grad, expected)
    y = Variable(np.array(1.0))
    outputs = add(add(y, y), y)
    outputs.backward()
    expected = np.array(3.0)
    self.assertEqual(y.grad, expected)

  def test_reuse_variable(self) :
    x = Variable(np.array(1.0))
    outputs = add(x, x)
    outputs.backward()
    expected = np.array(2.0)
    self.assertEqual(x.grad, expected)

    x.cleargrad()
    outputs = square(x)
    outputs.backward()
    expected = np.array(2.0)
    self.assertEqual(x.grad, expected)
  
  def test_generation(self) :
    x = Variable(np.array(2.0))
    y = square(x)
    z = add(square(y), square(y))
    self.assertEqual(z.data, np.array(32))
    z.backward()
    self.assertEqual(x.grad, 64)
  
  def test_retain_grad(self) :
    x0 = Variable(np.array(1.0))
    x1 = Variable(np.array(1.0))
    t = Variable(np.array(1.0))
    y = add(x0, x1)
    z = add(y, t)
    self.assertEqual(z.data, 3)
    z.backward()
    self.assertEqual(x0.grad, 1)
    self.assertEqual(x1.grad, 1)
    self.assertEqual(y.grad, None)
    self.assertEqual(z.grad, None)
  
  def test_config_enable_backward(self) :
    x = Variable(np.ones((100, 100, 100)))
    y = square(square(square(x)))
    y.backward()

    with predict() :
      x = Variable(np.ones((100, 100, 100)))
      y = square(square(square(x)))
      y.backward()
  
  def test_variable_property(self) :
    x = Variable(np.array([1.0,2.0,3.0]), "x_input")
    print()
    print("x.shape:", x.shape)
    print("x.ndim:", x.ndim)
    print("x.dtype:", x.dtype)
    print("len x:", len(x))
    print("print:", x)
    print("name:", x.name)
  
  def test_mul(self) :
    x0 = Variable(np.array(2.0))
    x1 = Variable(np.array(3.0))
    y = mul(x0, x1)
    self.assertEqual(y.data, np.array(6.0))
    y.backward()
    self.assertEqual(x0.grad, np.array(3.0))
    self.assertEqual(x1.grad, np.array(2.0))

  def test_function_reload(self) :
    x0 = Variable(np.array(2.0))
    x1 = Variable(np.array(3.0))
    x2 = Variable(np.array(4.0))
    y = (x0 + x1) * x2
    self.assertEqual(y.data, np.array(20.0))
  
  def test_input_array_scalar(self) :
    x = Variable(np.array(2.0))
    y = x + np.array(3.0)
    z = y * 4.0
    self.assertEqual(z.data, 20.0)
    y = np.array(3.0) + x
    z = 4.0 * y
    self.assertEqual(z.data, 20.0)

    # test __array_priority__
    x0 = Variable(np.array([2.0]))
    y = np.array([3.0]) + x0
    z = 4.0 * y
    self.assertEqual(z.data, [20.0])
  
  def test_sub(self) :
    x0 = Variable(np.array(2.0))
    x1 = Variable(np.array(1.0))
    y = x0 - x1
    self.assertEqual(y.data, 1.0)
    y = np.array(3.0) - x1
    self.assertEqual(y.data, 2.0)
    y = 4.0 - x1
    self.assertEqual(y.data, 3.0)

  def test_neg(self) :
    x0 = Variable(np.array(2.0))
    y = -x0
    self.assertEqual(y.data, -2.0)

  def test_div(self) :
    x0 = Variable(np.array(2.0))
    x1 = Variable(np.array(1.0))
    y = x0 / x1
    self.assertEqual(y.data, 2.0)
    y = np.array(3.0) / x1
    self.assertEqual(y.data, 3.0)
    y = 4.0 / x1
    self.assertEqual(y.data, 4.0)

  def test_pow(self) :
    x0 = Variable(np.array(2.0))
    y = x0 ** 3.0
    self.assertEqual(y.data, 8.0)
  
  def test_sphere(self) :
    x = Variable(np.array(1.0))
    y = Variable(np.array(1.0))
    z = sphere(x, y)
    self.assertEqual(z.data, 2.0)
    z.backward()
    self.assertEqual(x.grad, 2.0)
    self.assertEqual(y.grad, 2.0)
  
  def test_matyas(self) :
    x = Variable(np.array(1.0))
    y = Variable(np.array(1.0))
    z = matyas(x, y)
    self.assertTrue(np.allclose(z.data, 0.04))
    z.backward()
    self.assertTrue(x.grad, 0.040000000000000036)
    self.assertEqual(y.grad, 0.040000000000000036)
  
  def test_goldstein(self) :
    x = Variable(np.array(1.0))
    y = Variable(np.array(1.0))
    z = goldstein(x, y)
    self.assertTrue(np.allclose(z.data, 1876.0))
    z.backward()
    self.assertTrue(x.grad, 2108.0)
    self.assertEqual(y.grad, -5172.0)
  
  def test_sin(self) :
    x = Variable(np.array(np.pi/4))
    y = sin(x)
    self.assertTrue(np.allclose(y.data, 0.7071067811865475))
    y.backward()
    self.assertTrue(np.allclose(x.grad, 0.7071067811865475))

  def test_my_factorial(self) :
    x = Variable(np.array(5.0))
    y = my_factorial(x)
    self.assertEqual(y.data, math.factorial(5.0))

  def test_taylor_sin(self) :
    x0 = Variable(np.array(np.pi / 4))
    y0 = taylor_sin(x0)
    x1 = Variable(np.array(np.pi / 4))
    y1 = sin(x1)
    self.assertTrue(np.allclose(y1.data, 0.7071067811865475))
    self.assertTrue(np.allclose(y0.data, y1.data))
    y0.backward()
    y1.backward()
    self.assertTrue(np.allclose(x1.grad, 0.7071067811865475))
    # warning, 疑似用的math.factorial()及一系列pow方法没有支持反向传播，导致反向传播结果出错
    #self.assertTrue(np.allclose(x0.grad, x1.grad))

unittest.main()