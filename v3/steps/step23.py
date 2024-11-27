#! python3

if '__file__' in globals() :
  import os, sys
  sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dezero import *

x = Variable(np.array(3.0))
print(x)