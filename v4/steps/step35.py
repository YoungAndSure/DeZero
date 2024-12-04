#! python3

if '__file__' in globals() :
  import os, sys
  sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dezero import *

x = Variable(np.array(1.0))
y = tanh(x)
x.name = "x"
y.name = "y"
y.backward(create_graph=True)

iter = 7
for i in range(iter) :
  gx = x.grad
  x.cleargrad()
  gx.backward(create_graph=True)

gx = x.grad
gx.name = "gx" + str(iter+1)
plot_dot_graph(gx, verbose=False, to_file='tanh.png')