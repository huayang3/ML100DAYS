# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 14:36:05 2020

@author: hua_yang
"""

cur_x = 3 # The algorithm starts at x=3
lr = 0.01 # Learning rate
precision = 0.000001 #This tells us when to stop the algorithm
previous_step_size = 1 #
max_iters = 10000 # maximum number of iterations
iters = 0 #iteration counter
df = lambda x: 2*(x+5) #Gradient of our function 

iters_history = [iters]
x_history = [cur_x]

while previous_step_size > precision and iters < max_iters:
    prev_x = cur_x #Store current x value in prev_x
    cur_x = cur_x - lr * df(prev_x) #Gradient descent
    previous_step_size = abs(cur_x - prev_x) # 取較大的值, Change in x
    iters = iters+1 #iteration count
    print("Iteration",iters,"\nX value is",cur_x) #Print iterations
     # Store parameters for plotting
    iters_history.append(iters)
    x_history.append(cur_x)

print("Totally iteations: ", iters)
print("The local minimum occurs at", cur_x)

import matplotlib.pyplot as plt
#%matplotlib inline 
#適用於 Jupyter Notebook, 宣告直接在cell 內印出執行結果

plt.plot(iters_history, x_history, 'o-', ms=3, lw=1.5, color='black')
plt.xlabel(r'$iters$', fontsize=16)
plt.ylabel(r'$x$', fontsize=16)
plt.show()