# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 18:54:10 2020

@author: hua_yang
"""

import numpy as np
 
'''
定義數學函數:
y=f(x) = 1/(1+np.exp(-x));
dy/dx = df = x*(1-x) 

利用 deriv (derivative)做變數, 來分別指定方程式

'''    
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
 
    return 1/(1+np.exp(-x))

# 參數定義
# X 輸入資料集，形式為矩陣，每 1 行代表 1 個訓練樣本。
# y 輸出資料集，形式為矩陣，每 1 行代表 1 個訓練樣本。

X = np.array([[0,0,1],
            [0,1,1],
            [1,0,1],
            [1,1,1]])
 
y = np.array([[0],
            [1],
            [1],
            [0]])
 
np.random.seed(1)
 
# randomly initialize our weights with mean 0 
# syn0 第一層權值
# syn1 第二層權值

syn0 = 2*np.random.random((3,4)) - 1
syn1 = 2*np.random.random((4,1)) - 1

syn0_history = [syn0]
syn1_history = [syn1]
 
for j in range(60000):
 
    # Feed forward through layers 0, 1, and 2
    # l0 網路第 1 層，即網路輸入層。
    # l1 網路第 2 層，常稱作隱藏層。
    # l2 假定為網路最後一層，隨著訓練進行，其輸出應該逐漸接近正確結果
    
    l0 = X
    l1 = nonlin(np.dot(l0,syn0))
    l2 = nonlin(np.dot(l1,syn1))
 
    # 如何計算損失: 
    # l1_error 該值為 l2_delta 經 syn1 加權後的結果，從而能夠計算得到中間層/隱層的損失。
    # l1_delta 該值為經確信度加權後的神經網路 l1 層的損失，除了確信損失很小時，它近似等於 l1_error 。
    

    l2_error = y - l2
    
    if (j% 10000) == 0:
        print("L2_Error:" + str(np.mean(np.abs(l2_error))))
 
    l2_delta = l2_error * nonlin(l2,True)
    # how much did each l1 value contribute to the l2 error (according to the weights)?
    l1_error = l2_delta.dot(syn1.T)
    
    if (j% 10000) == 0:
        print("L1_Error:" + str(np.mean(np.abs(l1_error))))
    # in what direction is the target l1?
    # were we really sure? if so, don't change too much.
    l1_delta = l1_error * nonlin(l1,deriv=True)
 
    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)
    
    syn1_history.append(syn1)
    syn0_history.append(syn0)
    
import matplotlib.pyplot as plt

plt.plot(syn0_history[0], ms=1, lw=1.5, color='black')
plt.xlabel(r'$L1$', fontsize=16)
plt.show()

plt.plot(syn1_history[0], 'o-', ms=1, lw=1.5, color='pink')
plt.xlabel(r'$L2$', fontsize=16)
plt.show()

print("Output After Training (根據 L1):")
print(l1)
print("Output After Training (根據 L2):")
print(l2)