import h5py
import numpy as np
from sklearn.utils import shuffle

np.random.seed(2017)

from keras.models import *
from keras.layers import *

def make_model(input_shape):
    input_tensor = Input(shape=input_shape)
    # x = Flatten()(inp)
    # x = Dropout(0.5)(input_tensor)
    # x = Dense(10, activation='softmax')(x)
    # model = Model(input_tensor, x)
    # model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    input_tensor = Input(input_shape)
    x = Dropout(0.5)(input_tensor)
    x = Dense(10, activation='softmax')(x)
    model = Model(input_tensor, x)
    model.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])

    return model

def one_hot_encode(y):
    l = list()
    for item in y:
        c = [0. for i in range(10)]
        c[item] = 1.
        l.append(c)
    return np.array(l)


import h5py
import numpy as np
from sklearn.utils import shuffle
np.random.seed(2017)

X_train = []
X_test = []

for filename in ["bottleneck_ResNet50.h5", "bottleneck_Xception.h5", "bottleneck_InceptionV3.h5"]:
    print('------------------'+filename)
    with h5py.File("models/" + filename, 'r') as h:
        X_train.append(np.array(h['train']))
        X_test.append(np.array(h['test']))
        y_train = np.array(h['label'])

X_train = np.concatenate(X_train, axis=1)
X_test = np.concatenate(X_test, axis=1)

X_train, y_train = shuffle(X_train, y_train)
#print(y_train[:100])
y_train = one_hot_encode(y_train)
#print(y_train[:100])


print(X_train.shape)
print(y_train.shape)

model = make_model(X_train.shape[1:])
model.fit(X_train, y_train, batch_size=128, epochs=30, validation_split=0.2)

