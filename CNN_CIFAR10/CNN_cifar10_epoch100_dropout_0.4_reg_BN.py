#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from keras.layers import Dropout
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Activation, Flatten, Dense, Conv2D, Convolution2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, Adadelta, SGD, RMSprop, Adagrad, Adamax, Nadam
from keras.backend import epsilon
import tensorflow as tf
from keras import regularizers
import matplotlib.pyplot as plt
import os, sys
from keras.regularizers import l2


def getModel():
    l2_alpha = 0.0005
    model = Sequential()

    weight_decay = 1e-4
    # Add as many layers
    for _ in range(5):
        model.add(Conv2D(64, kernel_size=3, activation="relu", input_shape=(32, 32, 3), kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))
        model.add(Conv2D(32, kernel_size=3, activation="relu", kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(10, activation="softmax"))

    return model


def saveModel(model):
    model.save(sys.argv[0].replace(".py",".h5"))#("CNN_cifar10.h5")
    model.save_weights(sys.argv[0].replace(".py","_weights.h5"))
    return


def train(model, epoch=1):
    optim = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss="categorical_crossentropy", optimizer=optim, metrics=["accuracy"])
    history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=epoch,batch_size=batch_size)
    saveModel(model)
    return history

print(len(cifar10.load_data()))

(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
# X_train=X_train.reshape(len(X_train),28,28,1)
Y_train = to_categorical(Y_train)
# X_test=X_test.reshape(len(X_test),28,28,1)
Y_test = to_categorical(Y_test)

epoch = 100
batch_size = 128

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto(allow_soft_placement=True)
tf.GPUOptions(per_process_gpu_memory_fraction=0.99)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

path = os.getcwd()

model = getModel()
model.summary()
history = train(model, epoch)
train_loss = history.history["loss"]
val_loss = history.history["val_loss"]
train_acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]

with open(sys.argv[0].replace(".py",".txt"), "w") as f:
    f.write("Epoch\t\t Train_Loss\t\t\t Val_loss\t\t\t Train_acc\t\t\t Val_acc\n")
    for ep in range(len(train_loss)):
        f.write(str(ep) + "\t\t " + str(train_loss[ep]) + "\t\t " + str(val_loss[ep]) + "\t\t " + str(
            train_acc[ep]) + "\t\t " + str(val_acc[ep]) + "\n")

