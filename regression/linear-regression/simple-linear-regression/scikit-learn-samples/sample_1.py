import matplotlib.pyplot as plot
import numpy as np
import pandas as pd
from sklearn import datasets, model_selection
from sklearn.linear_model import LinearRegression


#Set the features
data = datasets.load_boston();
df = pd.DataFrame(data.data, columns=data.feature_names)

#Set the target
target = pd.DataFrame(data.target, columns=["MEDV"])

#Extract one feature vs target variables
x = df["RM"]
y = target["MEDV"]

#model evaluation approach using train/test split
x_train, x_test, y_train, y_test = model_selection.train_test_split(x,y,test_size = .5)

#preprocess data
x_train = np.array(x_train).reshape((-1,1))
y_train = np.array(y_train).reshape((-1,1))
x_test = np.array(x_test).reshape((-1,1))
y_test = np.array(y_test).reshape((-1,1))

#choose ML Model
linearRegressor = LinearRegression()

#Model fitting
linearRegressor.fit(x_train,y_train, sample_weight=None)

#prediction
predicted = linearRegressor.predict(x_test)
expected = y_test

#print error rate
print("RMS: %r " % np.sqrt(np.mean((predicted - expected) ** 2)))

#evaluation
plot.scatter(expected, predicted)
plot.plot([0,50], [0,50], color = 'blue')
plot.title('MEDV vs RM (Training set)')
plot.xlabel('RM')
plot.ylabel('MEDV')
plot.show()





