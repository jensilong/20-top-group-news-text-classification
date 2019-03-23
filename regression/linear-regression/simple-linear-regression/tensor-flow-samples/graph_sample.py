import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = pd.read_csv('/Users/122287/PycharmProjects/Machine-Learning-Models/regression/linear-regression/simple-linear-regression/LinearRegressionSample1.csv', delimiter=',')

#Set dependent variable
x = data['X']

#Set independent variable
y = data['Y']

print(data.head())
print(x)
print(y)

plt.scatter(x,y, color='red')
plt.plot(x,y, color='blue')
plt.show()