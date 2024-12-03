#! python3
if '__file__' in globals() :
  import os, sys
  sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dezero import *
import matplotlib.pyplot as plt

count = 3
x = Variable(np.array(1.0))
y = sin(x)
y.backward(create_graph=True)
for i in range(count) :
  gx = x.grad
  x.cleargrad()
  gx.backward(create_graph=True)
  print(x.grad.data)

x = Variable(np.linspace(-7, 7, 200))
y = sin(x)
y.backward(create_graph=True)
logs = [y.data]
for i in range(3) :
  # 重申一遍x.grad含义，是在x.data位置的导数
  logs.append(x.grad.data)
  gx = x.grad
  x.cleargrad()
  gx.backward(create_graph=True)

labels = ["y=sin(x)", "y'", "y''", "y'''"]
for i, v in enumerate(logs) :
  plt.plot(x.data, logs[i], label=labels[i])
plt.legend(loc='lower right')
plt.show()