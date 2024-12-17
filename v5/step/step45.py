#! python3

if '__file__' in globals() :
  import os, sys
  sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import dezero as F
from dezero.layer import Layer,Linear
from dezero.model import Model
import numpy as np

np.random.seed(0)
x = np.random.rand(100, 1)

# test 1
model = Layer()
model.l1 = Linear(10)
model.l2 = Linear(1)

def predict(x, model) :
  y = model.l1(x)
  y = F.sigmod_simple(y)
  y = model.l2(y)
  return y

y = predict(x, model)
y.backward()
for param in model.params() :
  print("hi:", param)

model.cleargrad()

# test 2
class TwoLayerNet(Model) :
  def __init__(self, hidden_size, output_size) :
    super().__init__()
    self.l1 = Linear(out_size=hidden_size)
    self.l2 = Linear(out_size=output_size)

  def forward(self, x) :
    y = self.l1(x)
    y = F.sigmod_simple(y)
    y = self.l2(y)
    return y

x = F.Variable(np.random.randn(5, 10), name='x')
model = TwoLayerNet(100, 1)
model.plot(x)