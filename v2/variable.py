import numpy

class Variable :
    def __init__(self, data) :
        if (data is not None) :
            if (not isinstance(data, numpy.ndarray)) :
                raise TypeError('{} is not supported'.format(type(data)))
            self.data = data
        self.grad = None
        self.creator = None

    def backward(self) :
        if self.grad == None :
            self.grad = numpy.ones_like(self.data)
        funcs = [self.creator]
        while funcs :
            func = funcs.pop()
            output_datas = [output.grad for output in func.outputs]
            input_datas = func.backward(*output_datas)
            if not isinstance(input_datas, tuple) :
                input_datas = (input_datas, )
            for input, input_data in zip(func.inputs, input_datas) :
                input.grad = input_data
                if (input.creator != None) :
                    funcs.insert(0, input.creator)