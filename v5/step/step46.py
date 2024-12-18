#! python3

if '__file__' in globals() :
  import os, sys
  sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import dezero as C
import dezero.layer as L
import dezero.model as M
import dezero.optimizer as O
import numpy as np

np.random.seed(0)
x = np.random.rand(100, 1)
label_y = np.sin(2 * np.pi * x)

lr = 0.2

layer_size = (10, 1)
mlp_model = M.MLP(layer_size)
optimizer = O.Momentum(lr).setup(mlp_model)

iters = 10000
for i in range(iters) :
  predict_y = mlp_model.forward(x)
  loss = C.mean_square_error(predict_y, label_y)
  mlp_model.cleargrad()
  loss.backward()
  optimizer.update()

y = mlp_model.forward(x)
loss = C.mean_square_error(y, label_y)

loss_threshold = 1e-3
ok = (np.abs(loss.data) < loss_threshold).all()
print("ok:", ok)