#! python3

if '__file__' in globals() :
  import os, sys
  sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import dezero as C
import dezero.layer as L
import dezero.model as M
import dezero.optimizer as O
import external.datasets

import numpy as np
import math

x, t = external.datasets.get_spiral(train = True)

max_epoch = 300
batch_size = 30
lr = 1.0
hidden_size = 10

model = M.MLP(full_connect_layer_size=(hidden_size, 3))
optimizer = O.SDG(lr).setup(model)
data_size = len(x)
iter_size = math.ceil(data_size / batch_size)

for epoch in range(max_epoch) :
  index = np.random.permutation(data_size)
  loss_sum = 0

  for j in range(iter_size) :
    batch_start = j * batch_size
    batch_end = (j + 1) * batch_size
    batch_index = index[batch_start:batch_end]
    batch_x = x[batch_index]
    batch_t = t[batch_index]
    
    predict_y = model(batch_x)
    loss = C.softmax_cross_entropy_simple(predict_y, batch_t)
    model.cleargrad()
    loss.backward()
    optimizer.update()

    loss_sum += float(loss.data) * len(batch_t)
    print("epoch:{}, iter:{}, loss:{}".format(epoch, j, loss_sum))

print("loss={}".format(loss_sum/data_size))