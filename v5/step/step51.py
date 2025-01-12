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

train_set = E.MNIST()
test_set = E.MNIST(train = False)

batch_size = 100
train_loader = DL.DataLoader(train_set, batch_size)
test_loader = DL.DataLoader(test_set, batch_size, shuffle=False)

hidden_size = 1000
model = M.MLP((hidden_size, 10))

lr = 0.2
optimizer = O.SDG(lr).setup(model)

epoch_size = 5
for i in range(epoch_size) :
    sum_loss = 0
    sum_acc = 0
    for batch_x, batch_t in train_loader :
        predict_y = model(batch_x)
        acc = U.accuracy(predict_y, batch_t)
        loss = U.softmax_cross_entropy_simple(predict_y, batch_t)
        model.cleargrad()
        loss.backward()
        optimizer.update()

        sum_loss += loss.data * len(batch_x)
        sum_acc += acc.data * len(batch_x)
    print("epoch:{}, loss:{}, acc{}".format(i, sum_loss / len(train_set), sum_acc / len(train_set)))

with predict() :
    sum_loss = 0
    sum_acc = 0
    for batch_x, batch_t in test_loader :
        predict_y = model(batch_x)

        for i in range(len(batch_x)) :
            x = batch_x[i]
            plt.imshow(x.reshape(28, 28), cmap='gray')
            plt.axis('off')
            plt.title('label:{}'.format(predict_y[i]))
            plt.show()

        acc = U.accuracy(predict_y, batch_t)
        loss = U.softmax_cross_entropy_simple(predict_y, batch_t)
        sum_loss += loss.data * len(batch_x)
        sum_acc += acc.data * len(batch_x)
    print("loss:{}, acc{}".format(sum_loss / len(test_set), sum_acc / len(test_set)))
