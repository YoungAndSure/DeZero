#! python3

if '__file__' in globals() :
  import os, sys
  sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import dezero as C
import dezero.layer as L
import dezero.model as M
import numpy as np

np.random.seed(0)
x = np.random.rand(100, 1)
label_y = np.sin(2 * np.pi * x)# + np.random.rand(100, 1)

I,H,O = 1, 10, 1
class TwoLayerNet(M.Model) :
  def __init__(self, hidden_size, output_size) :
    super().__init__()
    self.l1 = L.Linear(out_size=hidden_size)
    self.l2 = L.Linear(out_size=output_size)
  
  def forward(self, x):
    y = self.l1(x)
    y = C.sigmod_simple(y)
    y = self.l2(y)
    return y

def predict(model, x) :
  y = model.forward(x)
  return y

iters = 10000
lr = 0.2
model = TwoLayerNet(H, O)

for i in range(iters) :
  predict_y = predict(model, x)
  loss = C.mean_square_error(predict_y, label_y)
  model.cleargrad()
  loss.backward()
  for param in model.params() :
    param.data -= lr * param.grad.data

y = predict(model, x)
loss = C.mean_square_error(y, label_y)

loss_threshold = 1e-3
ok = (np.abs(loss.data) < loss_threshold).all()
print("ok:", ok)