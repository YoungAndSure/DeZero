import numpy
from dezero.config import *

class Variable :
    def __init__(self, data, name=None) :
        if (data is not None) :
            if (not isinstance(data, numpy.ndarray)) :
                raise TypeError('{} is not supported'.format(type(data)))
            self.data = data
        self.grad = None
        self.creator = None
        self.generation = 0
        self.name = name

    @property 
    def shape(self) :
        return self.data.shape
    @property
    def ndim(self) :
        return self.data.ndim
    @property
    def dtype(self) :
        return self.data.dtype
    
    def __len__(self) :
        return len(self.data)
    
    def __repr__(self) :
        if self.data is None :
            return 'variable(None)'
        p = str(self.data)
        return 'variable(' + p + ')'

    def backward(self, retain_grad=False, create_graph=False) :
        if Config.enable_backward == False :
            #print("backward disabled")
            return

        if self.grad == None :
            self.grad = Variable(numpy.ones_like(self.data))

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
            output_grads = func.outputs().grad
            # 在反向传播时，可以认为实际上在执行一次新的正向传播，
            # 此时关掉enable_backward，也就关掉了这次正向传播对应的反向传播，
            # 对应第一次正向传播，也就是二次求导
            with using_config('enable_backward', create_graph) :
                input_grads = func.backward(output_grads)
                if not isinstance(input_grads, tuple) :
                    input_grads = (input_grads, )
                for input, input_grad in zip(func.inputs, input_grads) :
                    if (input.grad == None) :
                        input.grad = input_grad
                    else :
                        input.grad = input.grad + input_grad
                    if (input.creator != None) :
                        add_func(input.creator)
                if retain_grad == False :
                    func.outputs().grad = None

    def cleargrad(self) :
        self.grad = None

    #def reshape(self, *shape) :
    #    if len(shape) == 1 and isinstance(shape[0], (tuple, list)) :
    #        shape = shape[0]
    #    return dezero.function.reshape(self, shape)