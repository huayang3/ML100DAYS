# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 13:43:05 2020

@author: hua_yang
"""

from keras.layers import Input, Embedding, LSTM, Dense
from keras.models import Model

#主要輸入接收新聞標題本身，即一個整數序列（每個整數編碼一個詞）。
#這些整數在1 到10,000 之間（10,000 個詞的詞彙表），且序列長度為100 個詞
#宣告一個 NAME 去定義Input
main_input = Input(shape=(100,), dtype='int32', name='main_input')


# Embedding 層將輸入序列編碼為一個稠密向量的序列，
# 每個向量維度為 512。
x = Embedding(output_dim=512, input_dim=10000, input_length=100)(main_input)

# LSTM 層把向量序列轉換成單個向量，
# 它包含整個序列的上下文信息
lstm_out = LSTM(32)(x)

#插入輔助損失，使得即使在模型主損失很高的情況下，LSTM 層和Embedding 層都能被平穩地訓練
aux_output = Dense(1, activation='sigmoid', name='aux_out')(lstm_out)

#輔助輸入數據與LSTM 層的輸出連接起來，輸入到模型
import keras
aux_input = Input(shape=(5,), name='aux_in')
x = keras.layers.concatenate([lstm_out, aux_input])


# 堆疊多個全連接網路層
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)
#作業解答: 新增兩層
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)

# 最後添加主要的邏輯回歸層
main_output = Dense(1, activation='sigmoid', name='main_output')(x)



# 宣告 MODEL API, 分別採用自行定義的 Input/Output Layer
model = Model(inputs=[main_input, aux_input], outputs=[main_output, aux_output])

model.compile(optimizer='rmsprop',
              loss={'main_output': 'binary_crossentropy', 'aux_out': 'binary_crossentropy'},
              loss_weights={'main_output': 1., 'aux_out': 0.2})

model.summary()