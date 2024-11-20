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
            input = func.input
            input.grad = func.backward(func.output)
            if input.creator != None :
                funcs.insert(0, input.creator)