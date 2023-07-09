#gradient descent algorithm finds best fit line for training data 

import numpy as np

def gradient_descent(x,y):
    m_curr = b_curr = 0
    iterations = 10000 #initially keep low until learning rate is more or less optimized
    n = len(x)
    learning_rate = 0.08 #adjust as needed to get best fit for data, typically start at 0.0001/0.001, if it is too large you will pass global minimum and cost will increase

    for i in range(iterations):
        y_predicted = m_curr * x + b_curr
        cost = (1/n) * sum([val**2 for val in (y-y_predicted)]) #mean squared error/cost function (y-axis vs m and vs b, or in 3-D space vs both)
        md = -(2/n)*sum(x*(y-y_predicted)) #partial derivative slope function
        bd = -(2/n)*sum(y-y_predicted) #partial derivative y-intercept function
        m_curr = m_curr - learning_rate * md
        b_curr = b_curr - learning_rate * bd
        print ("m {}, b {}, cost {} iteration {}".format(m_curr,b_curr,cost, i))

x = np.array([1,2,3,4,5]) #data is training set/test data we are trying to fit a proper function to
y = np.array([5,7,9,11,13])

gradient_descent(x,y)
