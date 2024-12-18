

class Optimizer :
  def __init__(self) :
    self.target = None
    self.hooks = []

  def setup(self, target) :
    self.target = target
    return self

  def add_hook(self, hook) :
    self.hooks.append(hook)

  def update(self) :
    params = [param for param in self.target.params() if param.grad not None]

    # hook是批处理，处理所有参数
    for h in self.hooks() :
      h(params)

    # update_one的one意思是只处理一个param
    for param in params :
      self.update_one(param)

  def update_one(self, param) :
    raise NotImplementedError()

class SDG(Optimizer) :
  def __init__(self, lr=0.01) :
    super().__init__()
    self.lr = lr

  def update_one(self, param) :
    param.data -= self.lr * param.grad.data