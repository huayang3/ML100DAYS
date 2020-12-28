# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 15:07:44 2020

@author: hua_yang
"""

import numpy as np

# GRADED FUNCTION: conv_single_step
def conv_single_step(a_slice_prev, W, b):
    """
    定義一層 Kernel (內核), 使用的參數說明如下
    Arguments:
        a_slice_prev -- 輸入資料的維度
        W -- 權重, 被使用在 a_slice_prev
        b -- 偏差參數 
    Returns:
        Z -- 滑動窗口（W，b）卷積在前一個 feature map 上的結果
    """

    # 定義一個元素介於 a_slice and W
    s = a_slice_prev * W
    # 加總所有的 "s" 並指定給Z.
    Z = np.sum(s)
    # Add bias b to Z. 這是 float() 函數, 
    Z = float(Z + b)

    return Z

'''
seed( ) 用於指定隨機數生成時所用算法開始的整數值，
如果使用相同的seed( )值，則每次生成的隨即數都相同，
如果不設置這個值，則係統根據時間來自己選擇這個值，
此時每次生成的隨機數因時間差異而不同。
'''
np.random.seed(1)
#定義一個 4x4x3 的 feature map
a_slice_prev = np.random.randn(5, 5, 3)
W = np.random.randn(5, 5, 3)
b = np.random.randn(1, 1, 1)

#取得計算後,卷績矩陣的值
Z = conv_single_step(a_slice_prev, W, b)
print("Z =", Z)