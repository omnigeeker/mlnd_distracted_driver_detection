import os
import shutil

from keras.models import *
from keras.layers import *
from keras.applications import *
from keras.preprocessing.image import *

import h5py
import math

height = 598
width =1398

x = Input((height, width, 3))
vgg16_model = VGG16(input_tensor=x, weights='imagenet', include_top=False)
print(vgg16_model.summary())
index = 0
for layer in vgg16_model.layers:
    print(index, layer.name)
    index += 1

print(vgg16_model.get_layer("block4_pool").output)

# vgg19_model = VGG19(input_tensor=x, weights='imagenet', include_top=False)
# print(vgg16_model.summary())
#
# resnet50_model = ResNet50(input_tensor=x, weights='imagenet', include_top=False)
# print(resnet50_model.summary())
#
# inception_model = InceptionV3(input_tensor=x, weights='imagenet', include_top=False)
# print(inception_model.summary())
#
# xception_model = Xception(input_tensor=x, weights='imagenet', include_top=False)
# print(xception_model.summary())