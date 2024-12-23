import numpy as np
from external.datasets import get_spiral

class Dataset:
  def __init__(self, train = True, transforms=None, label_tansforms=None) :
    self.train = train
    self.data = None
    self.label = None
    self.transforms = transforms
    self.label_transforms = label_tansforms
    self.prepare()

  def __getitem__(self, index) :
    data = self.transforms(self.data[index]) if self.transforms != None else self.data[index]
    if self.label is None :
        return data, None
    label = self.label_transforms(self.label[index]) if self.label_transforms != None else self.label[index]
    return data, label
  
  def __len__(self) :
      return len(self.data)

  def prepare(self) :
      pass

class Spiral(Dataset) :
  def prepare(self) :
     self.data, self.label = get_spiral(train=True)

class BigData(Dataset) :
  def __getitem__(self, index) :
     x = np.load('data/{}.npy'.format(index))
     t = np.load('label/{}.npy'.format(index))
     return x,t

  def __len__(self) :
    # TODO:why?
    return 1000000