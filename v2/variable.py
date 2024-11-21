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
            output_grads = [output.grad for output in func.outputs]
            input_grads = func.backward(*output_grads)
            if not isinstance(input_grads, tuple) :
                input_grads = (input_grads, )
            for input, input_grad in zip(func.inputs, input_grads) :
                if (input.grad == None) :
                    input.grad = input_grad
                else :
                    input.grad = input.grad + input_grad
                if (input.creator != None) :
                    funcs.insert(0, input.creator)
    
    def cleargrad(self) :
        self.grad = None