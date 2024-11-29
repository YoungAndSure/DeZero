#! python3

if '__file__' in globals() :
  import os, sys
  sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dezero import *

x0 = Variable(np.array(0.0))
x1 = Variable(np.array(2.0))
lr = 0.001
iter = 50000

for i in range(iter) :
  print("iter:{} x0:{} x1:{}".format(i, x0.data, x1.data))
  y = rosenbrock(x0, x1)
  y.backward()
  x0.data -= x0.grad * lr
  x1.data -= x1.grad * lr
  x0.cleargrad()
  x1.cleargrad()

print(x0.data, x1.data)