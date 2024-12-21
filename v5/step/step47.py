#! python3

if '__file__' in globals() :
  import os, sys
  sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import dezero as C
import dezero.layer as L
import dezero.model as M
import dezero.optimizer as O

import numpy as np
import math

x = np.array([[0.2, -0.4], [0.3, 0.5], [1.3, -3.2], [2.1, 0.3]])
t = np.array([2, 0, 1, 0])
model = M.MLP((10, 3))
y = model(x)
loss = C.softmax_cross_entropy_simple(y, t)
print(loss)