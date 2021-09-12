#Gradient descent is an optimizing technique which uses hit and trial of intercept and slope to find lowest values.
#Gradient descent is an iterative optimization algorithm for finding the local minimum of a function.
#Cost function is a function that measures the performance of a model for any given data. Cost Function quantifies the
# error between predicted values and expected values and presents it in the form of a single real number.
# %%
import numpy as np 
# %%
def gradient_descent(x,y):
    m_curr = b_curr = 0
    iterations=10000
    n=len(x)
    learning_rate = 0.08
    
    for i in range(iterations):
        y_predicted = m_curr*x +b_curr
        cost=(1/n) * sum([val**2 for val in (y-y_predicted)])  #equation of cost
        md=-(2/n)*sum(x*(y-y_predicted)) #equation of gradient descent
        bd=-(2/n)*sum(y-y_predicted)   #equation2
        m_curr=m_curr-learning_rate*md
        b_curr=b_curr-learning_rate*bd
        print("m {} , b {}, cost {}, iterations {}".format(m_curr, b_curr, cost, i))
# %%
x=np.array([1,2,3,4,5])
# %%
y=np.array([5,7,9,11,13])
# %%
gradient_descent(x,y)
# %%
