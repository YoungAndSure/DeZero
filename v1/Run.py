#!/usr/bin/env python3

from Function import *
from Variable import Variable
import numpy as np


x = Variable(np.array(0.5))
y = square(exp(square(x)))
print(y.data)

y.backward()
print(x.grad)