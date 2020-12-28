# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 13:34:33 2020

@author: hua_yang
"""

#導入相關模組
import keras
from keras import layers
from keras import models
from keras.models import Sequential
from keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense

#確認keras 版本
print(keras.__version__)



model = Sequential([
    Dense(32, input_shape=(784,)),
    Activation('relu'),
    Dense(10),
    Activation('softmax'),
])

model.summary()


##建立一個序列模型
#model = models.Sequential()
##建立一個卷績層, 32 個內核, 內核大小 3x3, 
##輸入影像大小 28x28x1
#model.add(layers.Conv2D(32, (3, 3), input_shape=(28, 28, 1)))
#
##新增一池化層, 採用maxpooling
#model.add(MaxPooling2D(2,2))
#
##建立第二個卷績層, 池化層, 
##請注意, 不需要再輸入 input_shape
#model.add(layers.Conv2D(25, (3, 3)))
#model.add(MaxPooling2D(2,2))
#
##新增平坦層
#model.add(Flatten())
#
##建立一個全連接層
#model.add(Dense(units=100))
#model.add(Activation('relu'))
#
##建立一個輸出層, 並採用softmax
#model.add(Dense(units=10))
#model.add(Activation('softmax'))
#
##輸出模型的堆疊
#model.summary()