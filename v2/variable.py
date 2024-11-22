import numpy

class Variable :
    def __init__(self, data) :
        if (data is not None) :
            if (not isinstance(data, numpy.ndarray)) :
                raise TypeError('{} is not supported'.format(type(data)))
            self.data = data
        self.grad = None
        self.creator = None
        self.generation = 0

    def backward(self) :
        if self.grad == None :
            self.grad = numpy.ones_like(self.data)

        funcs = []
        seen_set = set()
        def add_func(f) :
            if f not in seen_set :
                funcs.append(f)
                seen_set.add(f)
            funcs.sort(key = lambda x : x.generation)
        add_func(self.creator)

        while funcs :
            func = funcs.pop()
            output_grads = [output().grad for output in func.outputs]
            input_grads = func.backward(*output_grads)
            if not isinstance(input_grads, tuple) :
                input_grads = (input_grads, )
            for input, input_grad in zip(func.inputs, input_grads) :
                if (input.grad == None) :
                    input.grad = input_grad
                else :
                    input.grad = input.grad + input_grad
                if (input.creator != None) :
                    add_func(input.creator)
    
    def cleargrad(self) :
        self.grad = None