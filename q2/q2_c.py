import sys
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras import backend as K

import matplotlib.pyplot as plt
from PIL import Image

from sklearn import svm
from sklearn.metrics import accuracy_score

import numpy as np
np.set_printoptions(threshold=np.inf)

def unpickle(file):
    import cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict

if "__name__" != "__main__":

    '''
    Each of the batch files contains a dictionary with the following elements:

    1. data -- a 10000x3072 numpy array of uint8s. Each row of the array stores 
       a 32x32 colour image. The first 1024 entries contain the red channel values,
       the next 1024 the green, and the final 1024 the blue. The image is stored
       in row-major order, so that the first 32 entries of the array are the red
       channel values of the first row of the image.
    2. labels -- a list of 10000 numbers in the range 0-9. The number at index i
       indicates the label of the ith image in the array data.

    The  batches.meta file contains a Python dictionary object. It has the label names. 
    '''

    batches = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"]
    fold_count = 5

    a = unpickle(sys.argv[1] + "/" + batches[0])
    x_train = a["data"]
    labels_train = a["labels"]
    b = unpickle("batches.meta")
    label_names = b["label_names"]

    for i in xrange(1, fold_count):
        a = unpickle(sys.argv[1] + "/" + batches[i])
        x_train = np.concatenate((x_train, a["data"]), axis = 0)
        labels_train.extend(a["labels"])
        b = unpickle("batches.meta")

    a = unpickle(sys.argv[2])
    x_test = a["data"]
    labels_test = a["labels"]

    x_train = np.reshape(x_train, (50000, 32, 32, 3), order='F')
    x_test = np.reshape(x_test, (10000, 32, 32, 3), order='F')

    input_shape = (32, 32, 3)
    epochs = 30

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(labels_train, num_classes=10)
    y_test = keras.utils.to_categorical(labels_test, num_classes=10)

    model = Sequential()
    model.add(Conv2D(64, kernel_size=(3, 3), input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('elu'))

    model.add(Conv2D(64, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('elu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('elu'))

    model.add(Conv2D(128, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('elu'))

    model.add(Flatten())

    model.add(Dense(64))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(Dropout(0.5))

    model.add(Dense(32))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(Dropout(0.5))

    model.add(Dense(10, activation='softmax'))

    opt = keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)
    # opt = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=opt,
                  metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=64, epochs=epochs, verbose=1,
              validation_data=(x_test, y_test))

    getFeatures = K.function([model.layers[0].input, K.learning_phase()],
                          [model.layers[19].output])

    train_features = []
    test_features = []

    for i in xrange(0, 40000, 50):
    # output in test mode
        train_features.extend(getFeatures([x_train[i:i+50, :, :, :], 0])[0])
    
    for i in xrange(0, 10000, 50):
    # output in test mode
        test_features.extend(getFeatures([x_test[i:i+50, :, :, :], 0])[0])

    train_features = np.asarray(train_features)
    test_features = np.asarray(test_features)

    # print train_features.shape
    # print test_features.shape

    SVM = svm.SVC()
    SVM.fit(train_features, labels_train)

    indices = SVM.predict(test_features)

    with open('q2_c_output.txt', 'w') as outfile:
        for i in indices:
            outfile.write(label_names[i-1]+'\n')

    # test_accuracy = accuracy_score(labels_test, SVM.predict(test_features))
    # print "Testing Accuracy: %.4f" % (test_accuracy)