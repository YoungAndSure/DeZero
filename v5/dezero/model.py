from dezero.layer import Layer
import dezero.util as util

class Model(Layer) :
  def plot(self, *input, file_name='model.png') :
    output = self.forward(*input)
    util.plot_dot_graph(output, verbose=True, to_file=file_name)