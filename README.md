# comp9417-machine-learning-and-data-mining
comp9417 machine learning and data mining notes and work

## Week 1: Regression

### 1.1 Supervised Learning
How to predict the house price given by its size?
1. Collect statistics -> Each house's preice and its size, then we have a table:
![table house price](https://github.com/yinyuWu/Coursera-ML-AndrewNg-Notes/blob/master/images/44c68412e65e62686a96ad16f278571f.png)
2. Find the relationship between these two variables, which means we need to find a function y = f(x) according to the table.

![learning](https://github.com/yinyuWu/Coursera-ML-AndrewNg-Notes/blob/master/images/2d99281dfc992452c9d32e022ce71161.png)

3. How do we know how good the function is? Cost function or MSE(mean squared error) to calculate the error(difference between predicted value and true value)

This is how Supervised Learning works:

![sl](https://github.com/yinyuWu/Coursera-ML-AndrewNg-Notes/blob/master/images/ad0718d6e5218be6e6fce9dc775a38e6.png)


### 1.2 Linear Regression
Predict a new value(y) according to a linear model.
Eg:
Predict 'house price' given the 'size of the house'

