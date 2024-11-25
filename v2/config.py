import contextlib

class Config :
  enable_backward = True

@contextlib.contextmanager
def using_config(key, value) :
  print("set " + key, value)
  old_value = getattr(Config, key)
  setattr(Config, key, value)
  try :
    yield
  finally :
    setattr(Config, key, old_value)

def predict() :
  return using_config("enable_backward", False)