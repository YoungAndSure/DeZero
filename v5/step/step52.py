#! python3

if '__file__' in globals() :
  import os, sys
  sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import dezero.core as C
import dezero.dataloaders as DL
import dezero.datasets as DS
import dezero.optimizer as O
import dezero.model as M
import dezero.user_defined_func as U
import external.datasets as E
from dezero.config import *
import matplotlib
import matplotlib.pyplot as plt

test_set = E.MNIST(train = False)

batch_size = 100
test_loader = DL.DataLoader(test_set, batch_size, shuffle=False)

hidden_size = 1000
model = M.MLP((hidden_size, 10))

path = "./model.npz"
if os.path.exists(path) :
   model.load_weights(path)

with predict() :
    sum_loss = 0
    sum_acc = 0
    for batch_x, batch_t in test_loader :
        predict_y = model(batch_x)
        acc = U.accuracy(predict_y, batch_t)
        loss = U.softmax_cross_entropy_simple(predict_y, batch_t)
        sum_loss += loss.data * len(batch_x)
        sum_acc += acc.data * len(batch_x)
    print("loss:{}, acc{}".format(sum_loss / len(test_set), sum_acc / len(test_set)))
