import math
import numpy as np

class DataLoader :
  def __init__(self, dataset, batch_size, shuffle=True) :
    self.data = dataset
    self.data_size = len(self.data)
    self.shuffle = shuffle
    self.batch_size = batch_size
    self.max_iter = math.ceil(len(self.data) / self.batch_size)
    self.iterate_count = 0

    self.reset()

  def __iter__(self) :
    return self

  def __next__(self) :
    if (self.iterate_count >= self.max_iter) :
      self.reset()
      raise StopIteration
    
    i = self.iterate_count
    batch_size = self.batch_size
    batch_index = self.index[i * batch_size: min((i + 1) * batch_size, self.data_size)]
    batch_data = [self.data[index] for index in batch_index]
    batch_x = [sample[0] for sample in batch_data]
    batch_t = [sample[1] for sample in batch_data]
    self.iterate_count += 1
    return np.array(batch_x), np.array(batch_t)

  def reset(self) :
    self.iterate_count = 0
    if self.shuffle :
      self.index = np.random.permutation(self.data_size)
    else :
      self.index = range(self.data_size)

  def next(self) :
    return self.__next__()