
import numpy as np
from variable import Variable

class Function :
    def __call__(self, *inputs) :
        # 上一个func输出是多个list，到这里就输入了多个list
        # 没明白官方怎么解决这个问题的，自己加了个展开操作代替之
        flat_inputs = self.flat_input(inputs)
        input_datas = [input.data for input in flat_inputs]
        self.generation = max([input.generation for input in flat_inputs])

        output_datas = self.forward(*input_datas)
        if not isinstance(output_datas, tuple) :
            output_datas = (output_datas, )
        outputs = []
        for output_data in output_datas :
            output = Variable(self.to_array(output_data))
            output.creator = self
            output.generation = self.generation + 1
            outputs.append(output)

        self.outputs = outputs
        self.inputs = flat_inputs
        return outputs

    def to_array(self, x) :
        if (np.isscalar(x)) :
            return np.array(x)
        return x
    
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

    # input: array, output: array
    def backward(self, x) :
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

class Square(Function) :
    def forward(self, x) :
        return x ** 2
    def backward(self, gy) :
        return 2 * self.inputs[0].data * gy
def square(*x) :
    f = Square()
    return f(*x)

class Exp(Function) :
    def forward(self, x) :
        return np.exp(x)
    def backward(self, gy) :
        return np.exp(self.inputs[0].data) * gy
def exp(*x) :
    f = Exp()
    return f(*x)