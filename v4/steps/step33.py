#! python3

if '__file__' in globals() :
  import os, sys
  sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dezero import *

x = Variable(np.array(2.0))
iter = 10

def f(x) :
  y = x ** 4 -2 * x ** 2
  return y

for i in range(iter) :
  print("iter:{} x:{}".format(i, x.data))
  y = f(x)
  y.backward(create_graph=True)

  gx = x.grad
  x.cleargrad()
  gx.backward()

  gx2 = x.grad

  x.data -= (gx.data / gx2.data)

  x.cleargrad()
  y.cleargrad()

print(x.data)