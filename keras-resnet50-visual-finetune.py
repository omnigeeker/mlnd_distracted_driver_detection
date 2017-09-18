import os
import cv2
import glob
import numpy as np
import pandas as pd

from keras.models import *
from keras.layers import *
from keras.applications import *
from keras.preprocessing.image import *

dir = "/ext/Data/distracted_driver_detection/"

model_image_size = 224

def lambda_func(x):
    x /= 255.
    x -= 0.5
    x *= 2
    return x

train_gen = ImageDataGenerator(
    featurewise_std_normalization=True,
    samplewise_std_normalization=True,
    rotation_range=10.,
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=0.1,
    zoom_range=0.1,
)
gen = ImageDataGenerator()

batch_size = 128
train_generator = train_gen.flow_from_directory(os.path.join(dir, 'train'), (model_image_size, model_image_size),shuffle=True, batch_size=batch_size, class_mode="categorical")
print("subdior to train type {}".format(train_generator.class_indices))
valid_generator = gen.flow_from_directory(os.path.join(dir, 'valid'), (model_image_size, model_image_size), shuffle=True, batch_size=batch_size, class_mode="categorical")
print("subdior to valid type {}".format(valid_generator.class_indices))

input_tensor = Input((model_image_size, model_image_size, 3))
x = input_tensor
if lambda_func:
    x = Lambda(lambda_func)(x)

base_model = ResNet50(input_tensor=Input((model_image_size, model_image_size, 3)), weights='imagenet', include_top=False)

x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.25)(x)
x = Dense(10, activation='softmax')(x)
model = Model(base_model.input, x)

z = zip([x.name for x in base_model.layers], range(len(base_model.layers)))
for k, v in z:
    print("{} - {}".format(k,v))

for i in range(140):
    model.layers[i].trainable = False

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#model.summary()
#model.fit(X_train, y_train, batch_size=16, epochs=10, validation_data=(X_valid, y_valid))
#model.fit_generator(train_generator, steps_per_epoch=10000, epochs=10)

steps_train_sample = 20787 // 128
steps_valid_sample = 1637 // 128

# model.fit_generator(train_generator,
#                     samples_per_epoch=steps_train_sample,
#                     validation_data=valid_generator,
#                     nb_val_samples=steps_valid_sample,
#                     nb_epoch=10)

model.fit_generator(train_generator, steps_per_epoch=steps_train_sample, epochs=10, validation_data=valid_generator, validation_steps=steps_valid_sample)

model.save("models/resnet50-imagenet-finetune140.h5")





