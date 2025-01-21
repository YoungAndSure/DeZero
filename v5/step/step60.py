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

max_epoch = 100
hidden_size = 100
out_size = 1
bptt_length = 30
batch_size = 30

train_set = SinCurve(train=True)
train_loader = DL.SeqDataLoader(train_set, batch_size=batch_size, shuffle=False)
seq_len = len(train_set)

model = M.BetterRNN(hidden_size=hidden_size, out_size=out_size)
optimizer = Adam().setup(model)

for i in range(max_epoch) :
  model.reset_status()
  loss = 0
  count = 0

  for x, t in train_loader :
    y = model(x)
    # NOTE:RNN特色
    loss += mean_square_error(y, t)
    count += 1

    if count % bptt_length == 0 or count == seq_len :
      model.cleargrad()
      loss.backward()
      # TODO: unchain之后，图都连不起来了，还怎么predict?
      loss.unchain_backward()
      optimizer.update()
  avg_loss = sum(loss.data).data / (count * batch_size)
  print("epoch:{}, loss:{}".format(i, avg_loss))

xs = np.cos(np.linspace(0, np.pi * 4, 1000))
y_list = []
# TODO: 为什么要reset_status?试了下，不reset也没什么问题
model.reset_status()
with predict() :
  for x in xs :
    x = np.array(x).reshape(1, 1)
    y = model(x)
    y_list.append(y.data.item())

plt.plot(np.arange(len(xs)), xs, label='y=cos(x)')
plt.plot(np.arange(len(xs)), y_list, label='predict')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()