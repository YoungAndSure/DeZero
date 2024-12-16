import dezero.core as Core
import numpy as np
import weakref

class Parameter(Core.Variable) :
    pass

class Layer :
    def __init__(self) :
        self._params = set()
    
    def __set_attr__(self, name, value) :
        if not isinstance(Parameter, value) :
            return
        self._params.add(name)
        super().__set_attr__(name, value)
    
    def __call__(self, *inputs) :
        outputs = self.forward(*inputs)
        if not isinstance(outputs, tuple) :
            outputs = (outputs,)
        self.inputs = [weakref.ref(input) for input in inputs]
        # Funcion里整个是虚的，因为你给阉割了，输出的outputs都是单个值
        self.outputs = [weakref.ref(output) for output in outputs]
        # DAMN : 原来这么简单，我直接阉割了Function，让它只支持输出一个output，来实现这个功能
        return outputs if len(outputs) > 1 else outputs[0]
 
    def forward(self) :
        raise NotImplementedError()

    def params(self) :
        # 为什么不直接遍历__dict__，而是非要存储个params？
        # 因为class的所有属性都会存在__dict__中，包括上边的inputs/outputs
        # 而用户只需要返回 params
        for name in self._params :
            yield self.__dict__[name]

    def cleargrad(self) :
        for param in self.params() :
            param.cleargrad()

class Linear(Layer) :
    def __init__(self, in_size, out_size, has_bias=True, dtype=np.float32) :
        super().__init__()
        I = in_size
        O = out_size
        W_data = np.random.rand(I, O).astype(dtype) / np.sqrt(1 / I)
        self.W = Parameter(W_data)
        if has_bias :
            self.b = Parameter(np.zeros(O, dtype=dtype))
        else :
            self.b = None

    def forward(self, x) :
        y = Core.linear(x, self.W, self.b)
        return y