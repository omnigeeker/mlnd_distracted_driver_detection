import os
import shutil

from keras.models import *
from keras.layers import *
from keras.applications import *
from keras.preprocessing.image import *

import h5py
import math

dir = "/ext/Data/distracted_driver_detection/"

def write_gap(tag, MODEL, image_size, lambda_func=None, featurewise_std_normalization=True):
    input_tensor = Input((*image_size, 3))
    x = input_tensor
    if lambda_func:
        x = Lambda(lambda_func)(x)
    base_model = MODEL(input_tensor=x, weights='imagenet', include_top=False)
    model = Model(base_model.input, GlobalAveragePooling2D()(base_model.output))

    print(MODEL.__name__)
    train_gen = ImageDataGenerator(
        featurewise_std_normalization=featurewise_std_normalization,
        samplewise_std_normalization=False,
        rotation_range=10.,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.1,
        zoom_range=0.1,
    )
    gen = ImageDataGenerator(
        featurewise_std_normalization=featurewise_std_normalization,
        samplewise_std_normalization=False,
    )

    batch_size = 16
    train_generator = train_gen.flow_from_directory(os.path.join(dir, 'train'), image_size, shuffle=False, batch_size=batch_size)
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
    with h5py.File(os.path.join("models", tag, "bottleneck_%s.h5") % MODEL.__name__) as h:
        h.create_dataset("train", data=train)
        h.create_dataset("valid", data=valid)
        h.create_dataset("label", data=train_generator.classes)
        h.create_dataset("valid_label", data=valid_generator.classes)
    print("write_gap {} successed".format(Model.__name__))

def write_gap_test(tag, MODEL, image_size, lambda_func=None, featurewise_std_normalization=True):
    input_tensor = Input((*image_size, 3))
    x = input_tensor
    if lambda_func:
        x = Lambda(lambda_func)(x)
    base_model = MODEL(input_tensor=x, weights='imagenet', include_top=False)
    model = Model(base_model.input, GlobalAveragePooling2D()(base_model.output))

    print(MODEL.__name__)
    gen = ImageDataGenerator(
        featurewise_std_normalization=featurewise_std_normalization,
        samplewise_std_normalization=False,
    )
    batch_size = 16
    test_generator = gen.flow_from_directory(os.path.join(dir, 'test'), image_size, shuffle=False, batch_size=batch_size, class_mode=None)
    print("predict_generator test {}".format(math.ceil(test_generator.samples/batch_size)))
    test = model.predict_generator(test_generator, math.ceil(test_generator.samples/batch_size))
    print("test: {}".format(test.shape))

    print("begin create database {}".format(Model.__name__))
    with h5py.File(os.path.join("models", tag, "bottleneck_%s_test.h5") % MODEL.__name__) as h:
        h.create_dataset("test", data=test)
    print("write_gap {} successed".format(Model.__name__))

def normal_preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2
    return x

###
### subdir = new1
###
if 0:
    print("===== Train & Valid =====")
    write_gap("new1", ResNet50, (224, 224), featurewise_std_normalization=False)
    write_gap("new1", Xception, (299, 299), xception.preprocess_input, featurewise_std_normalization=False)
    write_gap("new1", InceptionV3, (299, 299), inception_v3.preprocess_input, featurewise_std_normalization=False)
    write_gap("new1", VGG16, (224, 224), featurewise_std_normalization=False)
    write_gap("new1", VGG19, (224, 224), featurewise_std_normalization=False)

    print("===== Test =====")
    write_gap_test("new1", ResNet50, (224, 224), featurewise_std_normalization=False)
    write_gap_test("new1", Xception, (299, 299), xception.preprocess_input, featurewise_std_normalization=False)
    write_gap_test("new1", InceptionV3, (299, 299), inception_v3.preprocess_input, featurewise_std_normalization=False)
    write_gap_test("new1", VGG16, (224, 224), featurewise_std_normalization=False)
    write_gap_test("new1", VGG19, (224, 224), featurewise_std_normalization=False)

###
### subdir = new
###
if 0:
    print("===== Train & Valid =====")
    write_gap("new", ResNet50, (224, 224))
    write_gap("new", Xception, (299, 299), xception.preprocess_input)
    write_gap("new", InceptionV3, (299, 299), inception_v3.preprocess_input)
    write_gap("new", VGG16, (224, 224))
    write_gap("new", VGG19, (224, 224))

    print("===== Test =====")
    write_gap_test("new", ResNet50, (224, 224))
    write_gap_test("new", Xception, (299, 299), xception.preprocess_input)
    write_gap_test("new", InceptionV3, (299, 299), inception_v3.preprocess_input)
    write_gap_test("new", VGG16, (224, 224))
    write_gap_test("new", VGG19, (224, 224))

###
### subdir = noscale
###
if 1:
    print("===== Train & Valid =====")
    write_gap("noscale", ResNet50, (240, 320))
    write_gap("noscale", Xception, (360, 480), xception.preprocess_input)
    write_gap("noscale", InceptionV3, (360, 480), inception_v3.preprocess_input)
    write_gap("noscale", VGG16, (240, 320))
    write_gap("noscale", VGG19, (240, 320))

    print("===== Test =====")
    write_gap_test("noscale", ResNet50, (240, 320))
    write_gap_test("noscale", Xception, (360, 480), xception.preprocess_input)
    write_gap_test("noscale", InceptionV3, (360, 480), inception_v3.preprocess_input)
    write_gap_test("noscale", VGG16, (240, 320))
    write_gap_test("noscale", VGG19, (240, 320))
