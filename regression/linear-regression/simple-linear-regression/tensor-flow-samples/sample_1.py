import pandas as pd
import numpy as np
from sklearn import datasets, model_selection
import tensorflow as tf
import itertools

#Set the features
data = datasets.load_boston()
df = pd.DataFrame(data.data, columns=data.feature_names)

#Set the target
target = pd.DataFrame(data.target, columns=['MEDV'])

#Extract variables from df and target
x = df['RM']
y = target['MEDV']
x_train, y_train, x_test, y_test = model_selection.train_test_split(x,y, test_size=.5)

#preprocess data
x_train = np.array(x_train).reshape((-1,1))
y_train = np.array(y_train).reshape((-1,1))
x_test = np.array(x_test).reshape((-1,1))
y_test = np.array(y_test).reshape((-1,1))

#Create weight and bias, initialized to 0
w = tf.Variable(0.0, name="weights")
b = tf.Variable(0.0, name="bias")

#construct model to predict y
y_predicted = x * w + b

#Use the square error as the loss function
loss = tf.square(y - y_predicted, name="loss")

#Use gradient descent with learning rate of 0.01 to minimize loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

with tf.Session() as sess:

    #Initialize the necessary variables, in this case, w and b
    sess.run(tf.global_variables_initializer())

    #train the model
    for i in range(100): #run 100 epochs
        for a, b in data:
            # Session runs train_op to minimize loss
            sess.run(optimizer, feed_dict={x: a, y:b})

        #output the values of w and b
        w_value, b_value = sess.run([w,b])





