#! python3

if '__file__' in globals() :
  import os, sys
  sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import dezero as C
import dezero.layer as L
import numpy as np

np.random.seed(0)
x = np.random.rand(100, 1)
label_y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

I,H,O = 1, 10, 1
l1 = L.Linear(in_size = I, out_size = H)
l2 = L.Linear(in_size = H, out_size = O)

def predict(x) :
  y = l1(x)
  y = C.sigmod_simple(y)
  y = l2(y)
  return y

iters = 10000
lr = 0.2

for i in range(iters) :
  predict_y = predict(x)
  loss = C.mean_square_error(predict_y, label_y)
  l1.cleargrad()
  l2.cleargrad()

  loss.backward()

  for param in l1.params() :
    param.data -= lr * param.grad.data
  for param in l2.params() :
    param.data -= lr * param.grad.data

x_data = x.squeeze()
y_data = predict(x).data.squeeze()
print(label_y.squeeze())
print(y_data)