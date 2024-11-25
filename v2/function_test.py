#! python3

import unittest
import numpy as np

from variable import Variable
from function import *
from config import *

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

  def test_same_input(self) :
    x = Variable(np.array(1.0))
    outputs = add(x, x)
    outputs[0].backward()
    expected = np.array(2.0)
    self.assertEqual(x.grad, expected)
    y = Variable(np.array(1.0))
    outputs = add(add(y, y), y)
    outputs[0].backward()
    expected = np.array(3.0)
    self.assertEqual(y.grad, expected)

  def test_reuse_variable(self) :
    x = Variable(np.array(1.0))
    outputs = add(x, x)
    outputs[0].backward()
    expected = np.array(2.0)
    self.assertEqual(x.grad, expected)

    x.cleargrad()
    outputs = square(x)
    outputs[0].backward()
    expected = np.array(2.0)
    self.assertEqual(x.grad, expected)
  
  def test_generation(self) :
    x = Variable(np.array(2.0))
    y = square(x)
    z = add(square(y), square(y))
    self.assertEqual(z[0].data, np.array(32))
    z[0].backward()
    self.assertEqual(x.grad, 64)
  
  def test_retain_grad(self) :
    x0 = Variable(np.array(1.0))
    x1 = Variable(np.array(1.0))
    t = Variable(np.array(1.0))
    y = add(x0, x1)
    z = add(y, t)
    self.assertEqual(z[0].data, 3)
    z[0].backward()
    self.assertEqual(x0.grad, 1)
    self.assertEqual(x1.grad, 1)
    self.assertEqual(y[0].grad, None)
    self.assertEqual(z[0].grad, None)
  
  def test_config_enable_backward(self) :
    x = Variable(np.ones((100, 100, 100)))
    y = square(square(square(x)))
    y[0].backward()

    with predict() :
      x = Variable(np.ones((100, 100, 100)))
      y = square(square(square(x)))
      y[0].backward()
  
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
    self.assertEqual(y[0].data, np.array(6.0))
    y[0].backward()
    self.assertEqual(x0.grad, np.array(3.0))
    self.assertEqual(x1.grad, np.array(2.0))

  def test_function_reload(self) :
    x0 = Variable(np.array(2.0))
    x1 = Variable(np.array(3.0))
    x2 = Variable(np.array(4.0))
    y = (x0 + x1) * x2
    self.assertEqual(y[0].data, np.array(20.0))
 
unittest.main()