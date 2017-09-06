import os
import cv2
import glob
import numpy as np
from keras.models import *
from keras.layers import *
from keras.applications import *
from keras.preprocessing.image import *



basedir = "/ext/Data/distracted_driver_detection/"
subdir = "train"

model_image_size = 224

#n = 25000
#X = np.zeros((n, 224, 224, 3), dtype=np.uint8)
#y = np.zeros((n, 1), dtype=np.uint8)

X = list()
y = list()

for i in range(10):
    dir = os.path.join(basedir, subdir, "c%d"%i)
    image_files = glob.glob(os.path.join(dir,"*.jpg"))
    print("loding {}, image count={}".format(dir, len(image_files)))
    for image_file in image_files:
        image = cv2.imread(image_file)
        X.append(cv2.resize(image, (model_image_size, model_image_size)))
        label = np.zeros(10, dtype=np.uint8)
        label[i]=1
        y.append(label)

print("X size={} y size={}".format(len(X), len(y)))
print("X_shape:{} y_shape:{}".format(X[0].shape, y[0].shape))
