# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 18:31:17 2020

@author: hua_yang
"""

#def img_combine(img, ncols=8, size=1, path=False):
#    from math import ceil
#    import matplotlib.pyplot as plt
#    import numpy as np
#    nimg = len(img)
#    nrows = int(ceil(nimg/ncols))
#    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True, figsize=(ncols*size,nrows*size))
#    if nrows == 0:
#        return
#    elif ncols == 1:
#        for r, ax in zip(np.arange(nrows), axes):
#            nth=r
#            if nth < nimg:
#                ax.imshow(img[nth], cmap='rainbow', vmin=0, vmax=1)
#                
#            ax.set_axis_off()
#    elif nrows == 1:
#        for c, ax in zip(np.arange(ncols), axes):
#            nth=c
#            if nth < nimg:
#                ax.imshow(img[nth], cmap='rainbow', vmin=0, vmax=1)
#            ax.set_axis_off()
#    else:
#        for r, row in zip(np.arange(nrows), axes):
#            for c, ax in zip(np.arange(ncols), row):
#                nth=r*ncols+c
#                if nth < nimg:
#                    ax.imshow(img[nth], cmap='rainbow', vmin=0, vmax=1)
#                ax.set_axis_off()
#    plt.show()
#    
#from keras.datasets import cifar10
#(x_train, x_test), (y_train, y_test) = cifar10.load_data()
#
#def cifar_generator(image_array, batch_size=32):
#    while True:
#        for indexs in range(0, len(image_array), batch_size):
#            images = x_train[indexs: indexs+batch_size]
#            labels = x_test[indexs: indexs+batch_size]
#            yield images, labels
#
#
#cifar_gen = cifar_generator(x_train)
#images, labels = next(cifar_gen)
#img_combine(images)
#
#images, labels = next(cifar_gen)
#img_combine(images)

import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import RMSprop, Adam
import os

batch_size = 128 # batch 的大小，如果出現 OOM error，請降低這個值
num_classes = 10 # 類別的數量，Cifar 10 共有 10 個類別
epochs = 10 # 訓練的 epochs 數量

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

from sklearn.utils import shuffle
def my_generator(x, y, batch_size):
    while True:
        for idx in range(0, len(x), batch_size): # 讓 idx 從 0 開始，一次增加 batch size。假設 batch_size=32, idx = 0, 32, 64, 96, ....
            batch_x, batch_y = x[idx:idx+batch_size], y[idx:idx+batch_size]
            yield batch_x, batch_y
        x, y = shuffle(x, y) # loop 結束後，將資料順序打亂再重新循環
        
train_generator = my_generator(x_train, y_train, batch_size) # 建立好我們寫好的 generator

history = model.fit_generator(train_generator,
                    steps_per_epoch=int(len(x_train)/batch_size), # 一個 epochs 要執行幾次 update，通常是資料量除以 batch size
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

