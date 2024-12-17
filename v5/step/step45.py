#! python3

if '__file__' in globals() :
  import os, sys
  sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import dezero as F
import dezero.layer as L
import numpy as np

np.random.seed(0)
x = np.random.rand(100, 1)

model = L.Layer()
model.l1 = L.Linear(10)
model.l2 = L.Linear(1)

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