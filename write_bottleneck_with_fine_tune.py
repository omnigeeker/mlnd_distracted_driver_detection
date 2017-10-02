import os
import shutil

from keras.models import *
from keras.layers import *
from keras.applications import *
from keras.preprocessing.image import *

import h5py
import math

dir = "/ext/Data/distracted_driver_detection/"

resnet50_weight_file = "resnet50-imagenet-finetune152.h5"
xception_weight_file = "xception-imagenet-finetune116.h5"
inceptionV3_weight_file = "inceptionV3-imagenet-finetune172.h5"

def write_gap(tag, MODEL, weight_file, image_size, lambda_func=None, featurewise_std_normalization=True):
    input_tensor = Input((*image_size, 3))
    x = input_tensor
    if lambda_func:
        x = Lambda(lambda_func)(x)
    base_model = MODEL(input_tensor=x, weights=None, include_top=False)

    model = Model(base_model.input, GlobalAveragePooling2D()(base_model.output))
    model.load_weights("models/"+weight_file, by_name=True)

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

    batch_size = 64
    train_generator = train_gen.flow_from_directory(os.path.join(dir, 'train'), image_size, shuffle=False, batch_size=batch_size)
    print("subdior to train type {}".format(train_generator.class_indices))
    valid_generator = gen.flow_from_directory(os.path.join(dir, 'valid'), image_size, shuffle=False, batch_size=batch_size)
    print("subdior to valid type {}".format(valid_generator.class_indices))

    print("predict_generator train {}".format(math.ceil(train_generator.samples//batch_size+1)))
    train = model.predict_generator(train_generator, math.ceil(train_generator.samples//batch_size+1))
    print("train: {}".format(train.shape))
    print("predict_generator valid {}".format(math.ceil(valid_generator.samples//batch_size+1)))
    valid = model.predict_generator(valid_generator, math.ceil(valid_generator.samples//batch_size+1))
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

def write_gap_test(tag, MODEL, weight_file, image_size, lambda_func=None, featurewise_std_normalization=True):
    input_tensor = Input((*image_size, 3))
    x = input_tensor
    if lambda_func:
        x = Lambda(lambda_func)(x)
    base_model = MODEL(input_tensor=x, weights=None, include_top=False)
    model = Model(base_model.input, GlobalAveragePooling2D()(base_model.output))
    model.load_weights("models/" + weight_file, by_name=True)

    print(MODEL.__name__)
    gen = ImageDataGenerator(
        featurewise_std_normalization=featurewise_std_normalization,
        samplewise_std_normalization=False,
    )
    batch_size = 64
    test_generator = gen.flow_from_directory(os.path.join(dir, 'test'), image_size, shuffle=False, batch_size=batch_size, class_mode=None)
    print("predict_generator test {}".format(math.ceil(test_generator.samples//batch_size+1)))
    test = model.predict_generator(test_generator, math.ceil(test_generator.samples//batch_size+1))
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
### subdir = noscale
###
if 1:
    print("===== Train & Valid =====")
    write_gap("finetune", ResNet50, resnet50_weight_file, (240, 320))
    write_gap("finetune", Xception, xception_weight_file, (320, 480), xception.preprocess_input)
    write_gap("finetune", InceptionV3, inceptionV3_weight_file, (320, 480), inception_v3.preprocess_input)

    # print("===== Test =====")
    write_gap_test("finetune", ResNet50, resnet50_weight_file, (240, 320))
    write_gap_test("finetune", Xception, xception_weight_file, (320, 480), xception.preprocess_input)
    write_gap_test("finetune", InceptionV3, inceptionV3_weight_file, (320, 480), inception_v3.preprocess_input)
