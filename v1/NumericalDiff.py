#!/usr/bin/env python3

from Function import *

def numerical_diff(f, x, eps=1e-4) :
    y1 = f(Variable(x.data - eps))
    y2 = f(Variable(x.data + eps))
    return Variable((y2.data - y1.data) / (2 * eps))

def my_func(x) :
    square = Square()
    exp = Exp()
    return square(exp(square(x)))

y = numerical_diff(Square(), Variable(np.array(2.0)))
print(y.data)
y = numerical_diff(my_func, Variable(np.array(0.5)))
print(y.data)
