#The following is a simple linear regression exercise in python using sklearn's Linear Regression model class
#BLOCK 1
#Generate 2 lists x and y where x is 1,2,3,4,5... and y is 500, 505, 510, 515... by using the linear equation y=5x+500
import matplotlib.pyplot as plt

x=[]
y=[]
for i in range(100):
    x.append(i)
    y.append(i*5+500)
plt.scatter(x,y)
plt.xlabel('x: integers from 1 to 100')
plt.ylabel('y: 5x+500')
plt.show()
print(x)
print(y)

#BLOCK 2
#Train a model to predict the linear equation used for generating the array and run predictions against a set of known values  
from sklearn.linear_model import LinearRegression 
import numpy as np

#using the linear regression model from sklearn
model = LinearRegression()
#converting to data types comprehensible by the model
#converting x to a 2d array 
xn=np.array(x).reshape((-1, 1))
#converting x to a 1d array 
yn=np.array(y)
#During the training process, the model uses an optimization algorithm to find the optimal parameters (also known as weights or coefficients) that best fit the training data. The optimization algorithm minimizes a cost function that measures the difference between the predicted outputs and the true outputs. The specific optimization algorithm used depends on the type of model and the chosen settings.
model.fit(xn, yn)
#Testing the values 10,20,30,40 and printing the predictions
predictions = model.predict(np.array([10,20,30,40]).reshape((-1, 1)))
print(predictions)

# Get the coefficients and intercept
coef = model.coef_
intercept = model.intercept_

# Print the equation
print(f'y = {coef[0]} * x + {intercept}')
