import dezero.layer as L
from dezero.user_defined_func import *
import dezero.util as util

class Model(L.Layer) :
  def plot(self, *input, file_name='model.png') :
    output = self.forward(*input)
    util.plot_dot_graph(output, verbose=True, to_file=file_name)

class MLP(Model) :
  def __init__(self, full_connect_layer_size, activation=sigmod_simple) :
    super().__init__()
    self.activation = activation
    self.layers=[]
    for (i, layer_size) in enumerate(full_connect_layer_size) :
      l = L.Linear(out_size = layer_size)
      # 必须得放到self._params里，遍历参数时候才能拿到，之前都是用self.xxx=x
      # 调用了=，默认调用了__setattr__把参数放进self._params里了
      # 这里是append到一个数组里了，所以必须手动调用setattr
      setattr(self, "layer_"+str(i), l)
      self.layers.append(l)

  def forward(self, x) :
    for l in self.layers[:-1] :
      x = l(x)
      x = self.activation(x)
    output_layer = self.layers[-1]
    y = output_layer(x)
    return y