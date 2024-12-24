import dezero.core as Core
from dezero.config import *
import numpy as np
import weakref

class Parameter(Core.Variable) :
    pass

class Layer :
    def __init__(self) :
        self._params = set()
    
    def __setattr__(self, name, value) :
        if isinstance(value, (Parameter, Layer)) :
            self._params.add(name)
        super().__setattr__(name, value)
    
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
            param = self.__dict__[name]
            if isinstance(param, Parameter) :
                yield param
            else :
                yield from param.params()

    def cleargrad(self) :
        for param in self.params() :
            param.cleargrad()

class Linear(Layer) :
    def __init__(self, out_size, has_bias=True, dtype=np.float32, in_size=None) :
        super().__init__()
        self.I = in_size
        self.O = out_size
        self.W = None
        self.has_bias = has_bias
        self.dtype = dtype
        if self.I != None :
            self._init_W()

    def _init_W(self) :
        if Config.close_random :
            W_data = np.ones((self.I, self.O)).astype(self.dtype) / np.sqrt(1 / self.I)
        else :
            W_data = np.random.randn(self.I, self.O).astype(self.dtype) / np.sqrt(1 / self.I)
        self.W = Parameter(W_data, name='W')
        if self.has_bias :
            self.b = Parameter(np.zeros(self.O, dtype=self.dtype), name='b')
        else :
            self.b = None

    def forward(self, x) :
        if self.W == None :
            self.I = x.shape[1]
            self._init_W()
        y = Core.linear(x, self.W, self.b)
        return y