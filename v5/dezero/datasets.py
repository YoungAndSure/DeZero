from external.datasets import get_spiral

class Dataset:
  def __init__(self, train = True) :
    self.train = train
    self.data = None
    self.label = None
    self.prepare()

  def __getitem__(self, index) :
    if self.label is None :
      return self.data[index], None
    return self.data[index], self.label[index]
  
  def __len__(self) :
      return len(self.data)

  def prepare(self) :
      pass

class Spiral(Dataset) :
  def prepare(self) :
     self.data, self.label = get_spiral(train=True)