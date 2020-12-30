# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 10:42:01 2020

@author: hua_yang
"""

import numpy as np
from numpy import *
import matplotlib.pylab as plt
#%matplotlib inline

#Sigmoid 數學函數表示方式
#sigmoid = lambda x: 1 / (1 + np.exp(-x))
def sigmoid(x):
    return (1 / (1 + np.exp(-x)))

#Sigmoid 微分
def dsigmoid(x):
    return (x * (1 - x))

# linespace generate an array from start and stop value
# with requested number of elements. Example 10 elements or 100 elements.
x = plt.linspace(-10,10,100)

# prepare the plot, associate the color r(ed) or b(lue) and the label 
plt.plot(x, sigmoid(x), 'b', label='linspace(-10,10,10)')

# Draw the grid line in background.
plt.grid()

# 顯現圖示的Title
plt.title('Sigmoid Function')

# 顯現 the Sigmoid formula
plt.text(4, 0.8, r'$\sigma(x)=\frac{1}{1+e^{-x}}$', fontsize=15)

#resize the X and Y axes
plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1))
plt.gca().yaxis.set_major_locator(plt.MultipleLocator(0.1))
 
# create the graph
plt.show()

#Softmax 數學函數表示方式
def softmax(x):
     return np.exp(x) / float(sum(np.exp(x)))

#x=np.arange(0,1.0,0.01)
x = plt.linspace(-5,5,100)

#resize the X and Y axes
plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1))
#plt.gca().yaxis.set_major_locator(plt.MultipleLocator(1))

#列印所有Softmax 值並輸出成一陣列
print(softmax(x))
plt.plot(x, softmax(x), 'r')
plt.show()


#寫出 ReLU & dReLU 一階導數
def ReLU(x):
    #print(abs(x))
    #print(x>0)
    #print(abs(x) * (x > 0))
    return abs(x) * (x > 0)

def dReLU(x):
    return (1 * (x > 0))

#ReLU(-1)
# linespace generate an array from start and stop value
# with requested number of elements.
x = plt.linspace(-10,10,100)

# prepare the plot, associate the color r(ed) or b(lue) and the label 
plt.plot(x, ReLU(x), 'r')
plt.plot(x, dReLU(x), 'b')


# Draw the grid line in background.
plt.grid()

# Title
plt.title('ReLU Function')

# write the ReLU formula
plt.text(0, 9, r'$f(x)= (abs(x) * (x > 0))$', fontsize=15)

# create the graph
plt.show()
