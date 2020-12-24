# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 09:47:05 2020

@author: hua_yang
"""

import os
import keras
os.environ["CUDA_VISIBLE_DEVICES"] = "" # 使用 CPU

import numpy as np
import cv2 # 載入 cv2 套件
import matplotlib.pyplot as plt

train, test = keras.datasets.cifar10.load_data()

x_train, y_train = train
x_test, y_test = test
y_train = y_train.astype(int)
y_test = y_test.astype(int)

x_train_histogram = []
x_test_histogram = []
#hist1=[]
# 對於所有訓練資料
for i in range(len(x_train)):
    chans = cv2.split(x_train[i]) # 把圖像的 3 個 channel 切分出來
    #hist1 = cv2.calcHist(chans[0], [0], None, [16], [0, 256]) # 切成 16 個 bin
    #hist1=hist1.flatten()
    # 對於所有 channel
    hist_feature = []
    for chan in chans:
        # 計算該 channel 的直方圖
        hist = cv2.calcHist([chan], [0], None, [16], [0, 256]) # 切成 16 個 bin
        hist_feature.extend(hist.flatten())
    # 把計算的直方圖特徵收集起來
    x_train_histogram.append(hist_feature)

# 對於所有測試資料也做一樣的處理
for i in range(len(x_test)):
    chans = cv2.split(x_test[i]) # 把圖像的 3 個 channel 切分出來
    # 對於所有 channel
    hist_feature = []
    for chan in chans:
        # 計算該 channel 的直方圖
        hist = cv2.calcHist([chan], [0], None, [16], [0, 256]) # 切成 16 個 bin
        hist_feature.extend(hist.flatten())
    x_test_histogram.append(hist_feature)

x_train_histogram = np.array(x_train_histogram)
x_test_histogram = np.array(x_test_histogram)

# SZ=20
bin_n = 16 # Number of bins


def hog(img):
    #print(type(img))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)
    bins = np.int32(bin_n*ang/(2*np.pi))    # quantizing binvalues in (0...16)
    #print(bins)
    bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
    mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)     # hist is a 64 bit vector
    return hist.astype(np.float32)

##hog(np.array(x_train[0]))
##print(type(img))
#img = cv2.cvtColor(np.array(x_train[0]), cv2.COLOR_RGB2GRAY)
#gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
#gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
#mag, ang = cv2.cartToPolar(gx, gy)
#bins = np.int32(bin_n*ang/(2*np.pi))    # quantizing binvalues in (0...16)
##print(bins)
#bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
#mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
#hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
#hist = np.hstack(hists)     # hist is a 64 bit vector
x_train_hog = np.array([hog(x) for x in x_train])
x_test_hog = np.array([hog(x) for x in x_test])

SVM_hist = cv2.ml.SVM_create()
SVM_hist.setKernel(cv2.ml.SVM_LINEAR)
SVM_hist.setGamma(5.383)
SVM_hist.setType(cv2.ml.SVM_C_SVC)
SVM_hist.setC(2.67)

#training
SVM_hist.train(x_train_histogram, cv2.ml.ROW_SAMPLE, y_train)

# prediction
_, y_hist_train = SVM_hist.predict(x_train_histogram)
_, y_hist_test = SVM_hist.predict(x_test_histogram)

SVM_hog = cv2.ml.SVM_create()
SVM_hog.setKernel(cv2.ml.SVM_LINEAR)
SVM_hog.setGamma(5.383)
SVM_hog.setType(cv2.ml.SVM_C_SVC)
SVM_hog.setC(2.67)

#training
SVM_hog.train(x_train_hog, cv2.ml.ROW_SAMPLE, y_train)

# prediction
_, y_hog_train = SVM_hog.predict(x_train_hog)
_, y_hog_test = SVM_hog.predict(x_test_hog)

print("-----Histogram result-----")
print("Training acc:", 100 * (y_hist_train == y_train).sum() / len(y_train), "%")
print("Testing acc:", 100 * (y_hist_test == y_test ).sum() / len(y_test), "%")

print("-----HOG result-----")
print("Training acc:", 100 * (y_hog_train == y_train).sum() / len(y_train), "%")
print("Testing acc:", 100 * (y_hog_test == y_test ).sum() / len(y_test), "%")