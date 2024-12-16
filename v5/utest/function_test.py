#! python3

if '__file__' in globals() :
  import os, sys
  sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import unittest
import numpy as np
import math

from dezero import *
import dezero.layer as L

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
    # 输入是一维的，输出就也是一维的，如果想转置后成一列，输入要写成二维的
    #self.assertTrue(np.array_equal(y.data, [[1.0], [2.0], [3.0]]))
    self.assertTrue(np.array_equal(y.data, [1.0, 2.0, 3.0]))
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

  def test_mean_square_error(self) :
    x0 = Variable(np.array([4,5,6]))
    x1 = Variable(np.array([7,8,9]))
    y = mean_square_error(x0, x1)
    self.assertTrue(np.array_equal(y.data, np.array([3.0, 3.0, 3.0])))
    y.backward()
    self.assertTrue(np.array_equal(x0.grad.data, [-2.0, -2.0, -2.0]))

  def test_linear(self) :
    x0 = Variable(np.ones((100, 1)))
    W0 = Variable(np.ones((1, 1)))
    b0 = Variable(np.ones((1,)))
    y0 = linear(x0, W0, b0)
    self.assertTrue(np.array_equal(y0.data, np.ones((100, 1)) + 1))
    y0.backward()

    x1 = Variable(np.ones((100, 1)))
    W1 = Variable(np.ones((1, 1)))
    b1 = Variable(np.ones((1,)))
    y1 = linear_simple(x1, W1, b1)
    self.assertTrue(np.array_equal(y1.data, np.ones((100, 1)) + 1))
    y1.backward()

    self.assertTrue(np.array_equal(W0.grad.data, W1.grad.data))
    self.assertTrue(np.array_equal(b0.grad.data, b1.grad.data))
    self.assertTrue(np.array_equal(x0.grad.data, x1.grad.data))

  def test_sigmod_simple(self) :
    x = Variable(np.array([0, 0, 0]))
    y = sigmod_simple(x)
    self.assertTrue(np.array_equal(y.data, [0.5, 0.5, 0.5]))
    y.backward()
    self.assertTrue(np.array_equal(x.grad.data, [0.25, 0.25, 0.25]))
  
  def test_linear_layer(self) :
    x0 = Variable(np.ones((100, 1)))
    layer = L.Linear(in_size = 1, out_size = 1, has_bias = True, dtype = np.float32)
    y0 = layer(x0)
    self.assertTrue(isinstance(y0, Variable))

  def test_linear_layer2(self) :
    np.random.seed(0)
    x = np.random.rand(100, 1)
    label_y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)
    print(label_y)

    iters = 1000
    lr = 0.1

    I,H,O = 1, 10, 1
    def use_layer() :
      l1 = L.Linear(I, H)
      l2 = L.Linear(H, O)

      def predict(x) :
        y = l1(x)
        y = sigmod_simple(y)
        y = l2(y)
        y = sigmod_simple(y)
        return y

      for i in range(iters) :
        predict_y = predict(x)
        loss = mean_square_error(predict_y, label_y)
        l1.cleargrad()
        l2.cleargrad()
        loss.backward()
        for param in l1.params() :
          param.data -= param.grad.data
        for param in l2.params() :
          param.data -= param.grad.data

      x_data = x.squeeze()
      y_data = predict(x).data.squeeze()
      print(y_data)
      return x_data, y_data

    def use_function() :
      W1 = Variable(np.random.rand(I, H))
      b1 = Variable(np.zeros(H))
      W2 = Variable(np.random.rand(H, O))
      b2 = Variable(np.zeros(O))

      def predict(x) :
        y = linear(x, W1, b1)
        y = sigmod_simple(y)
        y = linear(y, W2, b2)
        y = sigmod_simple(y)
        return y

      for i in range(iters) :
        predict_y = predict(x)
        loss = mean_square_error(predict_y, label_y)

        W1.cleargrad()
        b1.cleargrad()
        W2.cleargrad()
        b2.cleargrad()

        loss.backward()

        W1.data -= W1.grad.data
        b1.data -= b1.grad.data
        W2.data -= W2.grad.data
        b2.data -= b2.grad.data

      x_data = x.squeeze()
      y_data = predict(x).data.squeeze()
      print(y_data)
      return x_data, y_data
    
    use_layer_result = use_layer()
    use_function_result = use_function()
    self.assertEqual(len(use_layer_result), len(use_function_result))

unittest.main()