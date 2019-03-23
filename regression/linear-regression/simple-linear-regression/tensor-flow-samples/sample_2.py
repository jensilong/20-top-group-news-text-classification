import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Read data from csv file
data = data = pd.read_csv('/Users/122287/PycharmProjects/Machine-Learning-Models/regression/linear-regression/simple-linear-regression/LinearRegressionSample1.csv', delimiter=',')

#Assign data to corresponding variables
x = data['X']
y = data['Y']

#Create weight and bias, initialized 0
#Linear equation formula = y = b + w * x

b = tf.Variable(0.0, name='bias')
w = tf.Variable(1.0, name='weight')

#error
predicted_y = b + (x * w)
print(predicted_y)