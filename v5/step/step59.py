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
from external.utils import get_file
from external.datasets import SinCurve
from external.optimizers import Adam
from dezero.config import *
from dezero.functions_conv import *
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image

max_epoch = 30
hidden_size = 100
out_size = 1
bptt_length = 30

train_set = SinCurve(train=True)
seq_len = len(train_set)

model = M.SimpleRNN(hidden_size=hidden_size, out_size=out_size)
optimizer = Adam().setup(model)

for i in range(max_epoch) :
  model.reset_status()
  loss = 0
  count = 0

  for x, t in train_set :
    x = x.reshape(1, 1)
    y = model(x)
    # NOTE:RNN特色
    loss += mean_square_error(y, t)
    count += 1

    if count % bptt_length == 0 or count == seq_len :
      model.cleargrad()
      loss.backward()
      loss.unchain_backward()
      optimizer.update()
  avg_loss = loss.data.item() / count
  print("epoch:{}, loss:{}".format(i, avg_loss))