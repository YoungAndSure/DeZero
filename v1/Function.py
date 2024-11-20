
import numpy as np
from Variable import Variable

class Function :
    def __call__(self, input) :
        output_data = self.forward(input.data)
        output = Variable(self.to_array(output_data))
        output.creator = self
        self.output = output
        self.input = input
        return output

    def to_array(self, x) :
        if (np.isscalar(x)) :
            return np.array(x)
        return x

    
    # input: array, output: array or scalar
    def forward(self, x) :
        raise NotImplementedError()

    # input: array, output: array
    def backward(self, x) :
        raise NotImplementedError()

class Square(Function) :
    def forward(self, x) :
        return x ** 2
    def backward(self, gy) :
        return 2 * self.input.data * gy.grad
def square(x) :
    f = Square()
    return f(x)

class Exp(Function) :
    def forward(self, x) :
        return np.exp(x)
    def backward(self, gy) :
        return np.exp(self.input.data) * gy.grad
def exp(x) :
    f = Exp()
    return f(x)