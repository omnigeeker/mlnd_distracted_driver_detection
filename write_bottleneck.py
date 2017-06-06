import os
import shutil

from keras.models import *
from keras.layers import *
from keras.applications import *
from keras.preprocessing.image import *

import h5py
import math

dir = "/ext/Data/distracted_driver_detection/"

def write_gap(MODEL, image_size, lambda_func=None):
    width = image_size[0]
    height = image_size[1]
    input_tensor = Input((height, width, 3))
    x = input_tensor
    if lambda_func:
        x = Lambda(lambda_func)(x)
    base_model = MODEL(input_tensor=x, weights='imagenet', include_top=False)
    model = Model(base_model.input, GlobalAveragePooling2D()(base_model.output))

    print(MODEL.__name__)
    gen = ImageDataGenerator()
    batch_size = 16
    train_generator = gen.flow_from_directory(os.path.join(dir, 'train'), image_size, shuffle=False, batch_size=batch_size)
    print("subdior to train type {}".format(train_generator.class_indices))
    valid_generator = gen.flow_from_directory(os.path.join(dir, 'valid'), image_size, shuffle=False, batch_size=batch_size)
    print("subdior to valid type {}".format(valid_generator.class_indices))

    print("predict_generator train {}".format(math.ceil(train_generator.samples/batch_size)))
    train = model.predict_generator(train_generator, math.ceil(train_generator.samples/batch_size))
    print("train: {}".format(train.shape))
    print("predict_generator valid {}".format(math.ceil(valid_generator.samples/batch_size)))
    valid = model.predict_generator(valid_generator, math.ceil(valid_generator.samples/batch_size))
    print("valid: {}".format(valid.shape))
    print("train label: {}".format(train_generator.classes.shape))
    print("valid label: {}".format(valid_generator.classes.shape))

    print("begin create database {}".format(Model.__name__))
    with h5py.File("models/bottleneck_%s.h5" % MODEL.__name__) as h:
        h.create_dataset("train", data=train)
        h.create_dataset("valid", data=valid)
        h.create_dataset("label", data=train_generator.classes)
        h.create_dataset("valid_label", data=valid_generator.classes)
    print("write_gap {} successed".format(Model.__name__))

def write_gap_test(MODEL, image_size, lambda_func=None):
    width = image_size[0]
    height = image_size[1]
    input_tensor = Input((height, width, 3))
    x = input_tensor
    if lambda_func:
        x = Lambda(lambda_func)(x)
    base_model = MODEL(input_tensor=x, weights='imagenet', include_top=False)
    model = Model(base_model.input, GlobalAveragePooling2D()(base_model.output))

    print(MODEL.__name__)
    gen = ImageDataGenerator()
    batch_size = 16
    test_generator = gen.flow_from_directory(os.path.join(dir, 'test'), image_size, shuffle=False, batch_size=batch_size, class_mode=None)
    print("predict_generator test {}".format(math.ceil(test_generator.samples/batch_size)))
    test = model.predict_generator(test_generator, math.ceil(test_generator.samples/batch_size))
    print("test: {}".format(test.shape))

    print("begin create database {}".format(Model.__name__))
    with h5py.File("models/bottleneck_%s_test.h5" % MODEL.__name__) as h:
        h.create_dataset("test", data=test)
    print("write_gap {} successed".format(Model.__name__))

# write_gap(ResNet50, (224, 224))
# write_gap(Xception, (299, 299), xception.preprocess_input)
# write_gap(InceptionV3, (299, 299), inception_v3.preprocess_input)
# write_gap(VGG16, (224, 224))
# write_gap(VGG19, (224, 224))

write_gap_test(ResNet50, (224, 224))
write_gap_test(Xception, (299, 299), xception.preprocess_input)
write_gap_test(InceptionV3, (299, 299), inception_v3.preprocess_input)
write_gap_test(VGG16, (224, 224))
write_gap_test(VGG19, (224, 224))