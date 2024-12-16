#! python3

if '__file__' in globals() :
  import os, sys
  sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dezero import *
import matplotlib.pyplot as plt

np.random.seed(0)
x = np.random.rand(100, 1)
label_y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

I,H,O = 1, 10, 1
W1 = Variable(np.random.rand(I, H))
b1 = Variable(np.zeros(H))
W2 = Variable(np.random.rand(H, O))
b2 = Variable(np.zeros(O))

def predict(x) :
  y = linear(x, W1, b1)
  y = sigmod_simple(y)
  y = linear(y, W2, b2)
  y = sigmod_simple(y)
  return y

iters = 1000
lr = 0.01

for i in range(iters) :
  predict_y = predict(x)
  loss = mean_square_error(predict_y, label_y)

  W1.cleargrad()
  b1.cleargrad()
  W2.cleargrad()
  b2.cleargrad()

  loss.backward()

  W1.data -= W1.grad.data
  b1.data -= b1.grad.data
  W2.data -= W2.grad.data
  b2.data -= b2.grad.data

x_data = x.squeeze()
y_data = predict(x).data.squeeze()
print(x_data, y_data)
plt.ion()
plt.plot(x_data, y_data)

plt.title("Simple Curve Plot")
plt.xlabel("X Axis")
plt.ylabel("Y Axis")

# 显示图形
plt.show()
input("Press any key to exit...")