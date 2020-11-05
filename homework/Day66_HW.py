# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 18:39:02 2020

@author: hua_yang
"""

import keras
from keras import backend as K
from keras.layers import Layer

#from keras.utils import multi_gpu_model
#from keras.models import Model
#from keras.layers import Input, Dense
#
#
#a = Input(shape=(32,))
#b = Dense(32)(a)
#model = Model(inputs=a, outputs=b)
#
#config = model.get_config()
#print(config)

print(keras.backend)
#print(keras.fuzz factor)


#設定浮點運算值
K.set_floatx('float16')
print(K.floatx())

