
import numpy as np
from variable import Variable

class Function :
    def __call__(self, *inputs) :
        input_datas = [input.data for input in inputs]
        output_datas = self.forward(*input_datas)
        if not isinstance(output_datas, tuple) :
            output_datas = (output_datas, )
        outputs = []
        for output_data in output_datas :
            output = Variable(self.to_array(output_data))
            output.creator = self
            outputs.append(output)
        self.outputs = outputs
        self.inputs = inputs
        return outputs

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

class Add(Function) :
    def forward(self, x0, x1) :
        y = x0 + x1
        return y
def add(*inputs) :
    func = Add()
    return func(*inputs)