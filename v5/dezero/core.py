import numpy as np
import numpy
import weakref
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

    def reshape(self, *shape) :
        # *会把输入变成一个tuple，所以，如果传入(6,)，这里读出来是((6,))
        # 如果传入2,3，这里读出来反倒是(2,3)，可以直接透传
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)) :
            shape = shape[0]
        return reshape(self, shape)
    
    def transpose(self) :
        return transpose(self)
    @property
    def T(self) :
        return transpose(self)

def as_variable(input) :
    if not isinstance(input, Variable) :
        return Variable(input)
    return input

def as_array(x) :
    if (np.isscalar(x)) :
        return np.array(x)
    return x

class Function :
    def __call__(self, *inputs) :
        inputs = [as_variable(as_array(input)) for input in inputs]
        input_datas = [input.data for input in inputs]
        self.generation = max([input.generation for input in inputs])

        output_datas = self.forward(*input_datas)
        outputs = Variable(as_array(output_datas))
        if Config.enable_backward == True :
            outputs.creator = self
            outputs.generation = self.generation + 1

            # 真他妈巧妙，这里function持有的是虚的，向后传递的是实的，不用改动任何接口
            # 只有在用function里这个output的时候才需要加()
            self.outputs = weakref.ref(outputs)
            self.inputs = inputs
        return outputs

    def flat_input(self, inputs) :
        result = []
        for input in inputs :
            if (isinstance(input, list) or isinstance(input, tuple)) :
                result = result + input
            else :
                result.append(input)
        return result

    # 和上一版的类似，这两个接口是放给用户写的，所以务必完全适应用户的使用习惯，和系统的适配交由框架解决
    # 这里用户就想写个公式，并不想管几个输入和几个输出的问题，所以框架用两个技巧给解决了：
    # 一个是对输入解包，一个是对输出框架层转换
    # input: array, output: array or scalar
    def forward(self, x) :
        raise NotImplementedError()

    # input: Variable, output: Varibale
    def backward(self, gy) :
        raise NotImplementedError()

class Add(Function) :
    def forward(self, x0, x1) :
        y = x0 + x1
        return y
    def backward(self, gy) :
        return (gy, gy)
def add(*inputs) :
    func = Add()
    return func(*inputs)

class Sub(Function) :
    def forward(self, x0, x1) :
        y = x0 - x1
        return y
    def backward(self, gy) :
        return (gy, -1 * gy)
def sub(*inputs) :
    func = Sub()
    return func(*inputs)
def rsub(*inputs) :
    func = Sub()
    return func(inputs[1], inputs[0])

class Mul(Function) :
    def forward(self, x0, x1) :
        y = x0 * x1
        return y
    def backward(self, gy) :
        return (gy * self.inputs[1], gy * self.inputs[0])
def mul(*inputs) :
    func = Mul()
    return func(*inputs)

class Square(Function) :
    def forward(self, x) :
        return x ** 2
    def backward(self, gy) :
        return 2 * self.inputs[0] * gy
def square(*x) :
    f = Square()
    return f(*x)

class Exp(Function) :
    def forward(self, x) :
        return np.exp(x)
    def backward(self, gy) :
        return np.exp(self.inputs[0]) * gy
def exp(*x) :
    f = Exp()
    return f(*x)

class Neg(Function) :
    def forward(self, x) :
        return -1 * x
    def backward(self, gy) :
        return -1 * gy
def neg(*inputs) :
    func = Neg()
    return func(*inputs)

class Div(Function) :
    def forward(self, x0, x1) :
        return x0 / x1
    def backward(self, gy) :
        x0, x1 = self.inputs
        return (gy / x1, -1 * (x0 * gy) / (x1 * x1))
def div(*inputs) :
    func = Div()
    return func(*inputs)
def rdiv(*inputs) :
    func = Div()
    return func(inputs[1], inputs[0])

class Pow(Function) :
    def __init__(self, c) :
        self.c = c
    def forward(self, x) :
        return np.power(x, self.c)
    def backward(self, gy) :
        return gy * self.c * pow(self.inputs[0], self.c - 1)
def pow(x, c) :
    func = Pow(c)
    return func(x)

class Sin(Function) :
  def forward(self, x) :
    return np.sin(x)
  def backward(self, gy) :
    return cos(self.inputs[0]) * gy
def sin(x) :
  func = Sin()
  return func(x)

class Cos(Function) :
  def forward(self, x) :
    return np.cos(x)
  def backward(self, gy) :
    return -sin(self.inputs[0]) * gy
def cos(x) :
  func = Cos()
  return func(x)

class Tanh(Function) :
    def forward(self, x) :
        return np.tanh(x)
    def backward(self, gy) :
        return gy * (1 - self.outputs() ** 2)
def tanh(x) :
    func = Tanh()
    return func(x)

class Reshape(Function) :
    def __init__(self, shape) :
        self.output_shape = shape
    def forward(self, x) :
        self.input_shape = x.shape
        y = x.reshape(self.output_shape)
        return y
    def backward(self, gy) :
        return reshape(gy, self.input_shape)
def reshape(x, shape) :
    func = Reshape(shape)
    return func(x)

class Transpose(Function) :
    def forward(self, x) :
        return np.transpose(x)
    def backward(self, gy):
        # gy的类型是 Variable , 反向传播需要建立图，也就是要调用 DeZero 的方法实现
        return transpose(gy)
def transpose(x) :
    func = Transpose()
    return func(x)

def setup_variable() :
  Variable.__add__ = add
  Variable.__radd__ = add

  Variable.__mul__ = mul
  Variable.__rmul__ = mul

  Variable.__sub__ = sub
  Variable.__rsub__ = rsub

  Variable.__neg__ = neg

  Variable.__truediv__ = div
  Variable.__rtruediv__ = rdiv

  Variable.__pow__ = pow

  Variable.__array_priority__ = 200