
import numpy as np
from variable import Variable

class Function :
    def __call__(self, inputs) :
        input_datas = [input.data for input in inputs]
        output_datas = self.forward(input_datas)
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
    def forward(self, inputs) :
        x0, x1 = inputs[0], inputs[1]
        return [(x0 + x1)]
def add(inputs) :
    func = Add()
    return func(inputs)