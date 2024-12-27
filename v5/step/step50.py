#! python3

if '__file__' in globals() :
  import os, sys
  sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import dezero as C
import dezero.layer as L
import dezero.model as M
import dezero.optimizer as O
import dezero.datasets as D
import dezero.dataloaders as DL

import numpy as np
import math

dataset = D.Spiral(train=True)
max_epoch = 300
batch_size = 30
data_loader = DL.DataLoader(dataset, batch_size, shuffle=True)

lr = 1.0
hidden_size = 10
model = M.MLP(full_connect_layer_size=(hidden_size, 3))
optimizer = O.SDG(lr).setup(model)

for epoch in range(max_epoch) :
  loss_sum = 0
  for batch_x, batch_t in data_loader :
    predict_y = model(batch_x)
    loss = C.softmax_cross_entropy_simple(predict_y, batch_t)
    model.cleargrad()
    loss.backward(retain_grad=True)
    optimizer.update()
    loss_sum += float(loss.data) * len(batch_t)
  print('epoch %d, loss %.2f' % (epoch + 1, loss_sum / len(dataset)))