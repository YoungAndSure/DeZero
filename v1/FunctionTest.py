#! python3

import unittest
from Function import *

class SquareTest(unittest.TestCase) :
  def test_square(self) :
    x = Variable(np.array(2.0))
    y = square(x)
    expected = np.array(4.0)
    self.assertEqual(y.data, expected)
  
  def test_backward(self) :
    x = Variable(np.array(3.0))
    y = square(x)
    y.backward()
    expected = np.array(6.0)
    self.assertEqual(x.grad, expected)

class ExpTest(unittest.TestCase) :
  def test_exp(self) :
    x = Variable(np.array(2.0))
    y = exp(x)
    expected = np.array(np.exp(2.0))
    self.assertEqual(y.data, expected)

  def test_backward(self) :
    x = Variable(np.array(2.0))
    y = exp(x)
    y.backward()
    expected = np.array(y.data)
    self.assertEqual(x.grad, y.data)

class ComplicateTest(unittest.TestCase) :
  def to_array(self, x) :
    if (np.isscalar(x)) :
      return np.array(x)
  def numerical_diff(self, f, x, eps=1e-4) :
      y1 = f(Variable(self.to_array(x.data - eps)))
      y2 = f(Variable(self.to_array(x.data + eps)))
      result = self.to_array((y2.data - y1.data) / (2 * eps))
      return Variable(result)
  def test(self) :
    x = Variable(np.array(3.0))
    def test_func(x) :
      return square(exp(square(x)))
    num_diff = self.numerical_diff(test_func, x)
    y = test_func(x)
    y.backward()
    flg = np.allclose(x.grad, num_diff.data)
    self.assertTrue(flg)




unittest.main()