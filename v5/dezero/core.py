import numpy as np
import numpy
import weakref
from dezero.config import *
import dezero.util

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

  Variable.__getitem__ = get_item


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
            # THINK: 为什么这里output是虚的，input要是实的?
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
        self.x0_shape = x0.shape
        self.x1_shape = x1.shape
        y = x0 + x1
        return y
    def backward(self, gy) :
        gy0 = sum_to(gy, self.x0_shape)
        gy1 = sum_to(gy, self.x1_shape)
        return (gy0, gy1)
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
        return exp(self.inputs[0]) * gy
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
        # 不用特殊处理一维数据, 一维实际就是不能转置的
        #if x.ndim == 1 :
        #    y = np.transpose(x).reshape(-1, 1)
        #elif x.shape[-1] == 1 :
        #    y = np.transpose(x).ravel()
        #else :
        y = np.transpose(x)
        return y
        #return np.transpose(x) if x.ndim != 1 else np.transpose(x).reshape(-1, 1)
    def backward(self, gy):
        # gy的类型是 Variable , 反向传播需要建立图，也就是要调用 DeZero 的方法实现
        return transpose(gy)
def transpose(x) :
    func = Transpose()
    return func(x)

class Sum(Function) :
    def __init__(self, axis, keepdims) :
        self.axis = axis
        self.keepdims = keepdims
    def forward(self, x) :
        self.input_shape = x.shape
        return np.sum(x, axis=self.axis, keepdims=self.keepdims)
    def backward(self, gy) :
        # trick方法，这样反向传播没有用 dezero 实现的方法，会无法构建连接图，也就没法二次求导了
        #return Variable(np.ones(self.input_shape))
        # TODO:抄的人家的代码，没有特别明白
        gy = Utils.reshape_sum_backward(gy, self.input_shape, self.axis, self.keepdims)
        gy = broadcast_to(gy, self.input_shape)
        return gy
def sum(x, axis=None, keepdims=False) :
    func = Sum(axis, keepdims)
    return func(x)

class BroadcastTo(Function) :
    def __init__(self, output_shape) :
        self.output_shape = output_shape
    def forward(self, x) :
        self.input_shape = x.shape
        return np.broadcast_to(x, self.output_shape)
    def backward(self, gy) :
        return sum_to(gy, self.input_shape)
def broadcast_to(x, shape) :
    func = BroadcastTo(shape)
    return func(x)

class SumTo(Function) :
    def __init__(self, output_shape) :
        self.output_shape = output_shape
    def forward(self, x) :
        self.input_shape = x.shape
        return Utils.sum_to(x, self.output_shape)
    def backward(self, gy) :
        return broadcast_to(gy, self.input_shape)
def sum_to(x, shape) :
    func = SumTo(shape)
    return func(x)

class MatMul(Function) :
    def forward(self, x, W) :
        return np.dot(x, W)
    def backward(self, gy) :
        x = self.inputs[0]
        W = self.inputs[1]
        # 如果x是一维的，ndim == 1，x转置后就是二维的，ndim ==2
        # 而gy和y形状一样，是一维的,ndim == 1,没法和x.T相乘了,
        # 所以这里必须升一下维度，但是目前这么写没法处理三维的

        # 以上都不对，x和w的维度应该一致，不应该输入x是一维的，w是二维的，就会出现各种对不上的问题
        # 这段代码保留，理论上不会执行
        #if gy.data.ndim == 1 :
        #    gy.data = gy.data.reshape(1, len(gy.data))
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        return (gx, gW)
def matmul(x, W) :
    func = MatMul()
    return func(x, W)

class Log(Function) :
    def forward(self, x) :
        return np.log(x)
    def backward(self, gy) :
        return gy * (1 / self.inputs[0])
def log(x) :
    func = Log()
    return func(x)

class MeanSquareError(Function) :
    def forward(self, x0, x1) :
        diff = x0 - x1
        diff = diff ** 2
        self.N = len(diff.data)
        return diff / self.N
    def backward(self, gy) :
        x0 = self.inputs[0]
        x1 = self.inputs[1]
        gy0 = (2 / self.N) * (x0 - x1) * gy
        gy1 = (2 / self.N) * (x0 - x1) * -1 * gy
        return (gy0, gy1)
def mean_square_error(x0, x1) :
    func = MeanSquareError()
    return func(x0, x1)

class Linear(Function) :
    def forward(self, x, W, b=None) :
        # forward已经在一个Function里了，入出参都是np.array，不能再调用另一个Function
        # backward因为出入参是Variable类型，且需要建立反向传播的连接图，所以必须调用其他已经实现的Function
        t = np.dot(x, W)
        if b is None :
            return t
        self.b_shape = b.shape
        y = t + b
        return y
    def backward(self, gy) :
        # 入参是3个，返回梯度也是3个，一一对应
        gW = matmul(self.inputs[0].T, gy)
        gb = sum_to(gy, self.b_shape)
        gx = matmul(gy, self.inputs[1].T)
        return (gx, gW, gb)
def linear(*input) :
    func = Linear()
    return func(*input)

class GetItem(Function) :
    def __init__(self, slices) :
        # slices不能是个Variable，对slices求导数也没意义
        self.slices = slices
    def forward(self, x) :
        return x[self.slices]
    def backward(self, gy) :
        f = GetItemGrad(self.slices, self.inputs[0].shape)
        return f(gy)
class GetItemGrad(Function) :
    def __init__(self, slices, in_shape) :
        self.slices = slices
        self.in_shape = in_shape
    def forward(self, x) :
        y = np.zeros(self.in_shape)
        np.add.at(y, self.slices, x)
        return y
    def backward(self, gy) :
        return get_item(gy, self.slices)
def get_item(x, slices) :
    func = GetItem(slices)
    return func(x)

class Clip(Function) :
    def __init__(self, x_min, x_max) :
        self.x_min = x_min
        self.x_max = x_max
    def forward(self, x) :
        return np.clip(x, self.x_min, self.x_max)
    def backward(self, gy) :
        x = self.inputs[0]
        return gy * (x.data >= self.x_min) * (x.data <= self.x_max)
def clip(x, x_min, x_max) :
    func = Clip(x_min, x_max)
    return func(x)

class Utils :
    @staticmethod
    def sum_to(x, shape) :
        # copy from book example
        ndim = len(shape)
        lead = x.ndim - ndim
        lead_axis = tuple(range(lead))

        axis = tuple([i + lead for i, sx in enumerate(shape) if sx == 1])
        y = x.sum(lead_axis + axis, keepdims=True)
        if lead > 0:
            y = y.squeeze(lead_axis)
        return y

    @staticmethod
    def reshape_sum_backward(gy, input_shape, axis, keepdims) :
        # copy from book example
        ndim = len(input_shape)
        tupled_axis = axis
        if axis is None:
            tupled_axis = None
        elif not isinstance(axis, tuple):
            tupled_axis = (axis,)

        if not (ndim == 0 or tupled_axis is None or keepdims):
            actual_axis = [a if a >= 0 else a + ndim for a in tupled_axis]
            shape = list(gy.shape)
            for a in sorted(actual_axis):
                shape.insert(a, 1)
        else:
            shape = gy.shape

        gy = gy.reshape(shape)  # reshape
        return gy