#! python3

if '__file__' in globals() :
  import os, sys
  sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dezero import *

np.random.seed(0)
x = Variable(np.random.rand(100, 1))
label_y = 5 + 2 * x + np.random.rand(100, 1)

# W形状由x和y的形状决定，x是(100,1)，y是(100,1)，只有W是(1,1)才能点乘
W = Variable(np.zeros((1, 1)))
# b的形状实际和y是一样的，它会自动广播，所以这里设置为一维的
b = Variable(np.zeros((1)))

def predict(x) :
  y = matmul(x, W) + b
  return y

def mean_square_error(x0, x1) :
  diff = x0 - x1
  diff = diff ** 2
  return diff / len(diff.data)

lr = 0.1
for iter in range(100) :
  predict_y = predict(x)
  loss = mean_square_error(label_y, predict_y)

  W.cleargrad()
  b.cleargrad()
  loss.backward()

  W.data -= W.grad.data * lr
  b.data -= b.grad.data * lr
print(W.data, b.data)
print("loss:", loss)