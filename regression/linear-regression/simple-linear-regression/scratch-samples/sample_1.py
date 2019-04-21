import matplotlib.pyplot as plt



#set features
x = [1,2,2]

#set targets
y = [1,3,5]

#set initial values for linear regression
learning_rate = 0.0001
b = 0
w = 0
epoch = 300000

mse = set([])

def perform_error(x,y) :
	return sum([Y - ((X * w) + b) for X, Y in zip(x, y)])


def perform_gradient_descent(x , y , b, w,learning_rate,epoch):
	
	n = len(x)

	for i in range(epoch):
	
		w_gradient = -(2/n) * sum([X * (Y - ((X * w) + b)) for X, Y in zip(x, y)])
		b_gradient = -(2/n) * sum([Y - ((X * w) + b) for X, Y in zip(x,y)])
		cost = perform_error(x,y)
		mse.add(cost)
	
		w = w - (learning_rate * w_gradient)
		b = b - (learning_rate * b_gradient)
		
	return w , b, cost, mse

w , b , cost , mse = perform_gradient_descent(x,y,b,w,learning_rate,epoch)

y_predicted_list = []

for X in x :
	y_predicted = b + (X * w)
	y_predicted_list.append(y_predicted)


plt.scatter(x, y, color='red')
plt.plot(x,y, color='blue')
plt.plot(x, y_predicted_list, color='green')
plt.show()



