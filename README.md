# comp9417-machine-learning-and-data-mining
comp9417 machine learning and data mining notes and work

## Week 1: Regression

### 1.1 Supervised Learning
How to predict the house price given by its size?
1. Collect statistics -> Each house's preice and its size, then we have a table:
![table house price](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/44c68412e65e62686a96ad16f278571f.png)
2. Find the relationship between these two variables, which means we need to find a function y = f(x) according to the table.

![learning](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/8e76e65ca7098b74a2e9bc8e9577adfc.png)

3. How do we know how good the function is? Cost function or MSE(mean squared error) to calculate the error(difference between predicted value and true value)

This is how Supervised Learning works:

![sl](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/ad0718d6e5218be6e6fce9dc775a38e6.png)


### 1.2 Univariate Linear Regression
The core idea is to obtain a line that best fits the data. The best fit line is the one for which total prediction error (MSE) are as small as possible. Error is the distance between the point to the regression line.


![mse](https://i.imgur.com/vB3UAiH.jpg)

Eg:
Predict 'house price' given the 'size of the house', y = f(x) = b<sub>1</sub>x + b<sub>0</sub>(y=value of house, x=size of house).

What we need to find is b<sub>0</sub> and b<sub>1</sub>.

Coefficient Formula:

![coefficient](https://wikimedia.org/api/rest_v1/media/math/render/svg/8d2945202d09869511723ad4b0dfe5926cc3d2a0)

### 1.3 Multivariate Linear Regression

![multi](https://wikimedia.org/api/rest_v1/media/math/render/svg/8119b3ed1259aa8ff15166488548104b50a0f92e)

![multico](https://wikimedia.org/api/rest_v1/media/math/render/svg/6732e88e94d90d9e2ff8415882cb4ab1605790cb)
