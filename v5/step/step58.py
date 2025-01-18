#! python3

if '__file__' in globals() :
  import os, sys
  sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import dezero.core as C
import dezero.dataloaders as DL
import dezero.datasets as DS
import dezero.optimizer as O
import dezero.model as M
import dezero.user_defined_func as U
from external.utils import get_file
from dezero.config import *
from dezero.functions_conv import *
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image

url = 'https://github.com/oreilly-japan/deep-learning-from-scratch-3/' \
      'raw/images/zebra.jpg'
img_path = get_file(url)
img = Image.open(img_path)
x = M.VGG16.preprocess(img)
# 插入新轴，VGG输入是N,C,H,W,这里只有一张图片，是C,H,W，需要增加一个轴
x = x[np.newaxis]

model = M.VGG16(pretrained=True)
with test_mode() :
  y = model(x)
predict_id = np.argmax(y.data)

labels = DS.ImageNet.labels()
print(labels[predict_id])