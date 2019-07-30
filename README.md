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


## Week 2 Classification
### 2.1 Lazy Learners
Lazy learners simply store the training data and wait until a testing data appear. When it does, classification is conducted based on the most related data in the stored training data. Compared to eager learners, lazy learners have less training time but more time in predicting.

Ex. k-nearest neighbor, Case-based reasoning

### 2.2 Eager Learners
Eager learners construct a classification model based on the given training data before receiving data for classification. It must be able to commit to a single hypothesis that covers the entire instance space. Due to the model construction, eager learners take a long time for train and less time to predict.

Ex. Decision Tree, Naive Bayes, Artificial Neural Networks

### 2.3 K-nearest neighbor
In k-NN classification, the output is a class membership. An object is classified by a plurality vote of its neighbors, with the object being assigned to the class most common among its k nearest neighbors (k is a positive integer, typically small). If k = 1, then the object is simply assigned to the class of that single nearest neighbor

Thus, we need to find a formula to measure the distance.

Minkowski Distance:

![minkowski distance](https://wikimedia.org/api/rest_v1/media/math/render/svg/4060cc840aeab9e41b5e47356088889e2e7a6f0f)

The 2-norm refers to Euclidean Distance, and the 1-norm refers to Manhattan Distance. If p is infinitely large, distance is called Chebyshev distance.

### 2.4 Bayesian
h<sub>MAP</sub> = arg max(P(h|D))

![bay_map](https://wikimedia.org/api/rest_v1/media/math/render/svg/874397c7e77a6d91ce7e04550c62d3b582248f91)

Learning a real valued function: maximum likelihood hypothesis h<sub>ML</sub> is the one that minimises the sum of squared error.

![maxlikelyhood](https://cdn-images-1.medium.com/max/1600/1*t4zrihvhtlZJZsvcX3jRjg.png)

multivariate Bernoulli distribution:
For the i-th word in our vocabulary we have a random variable Xi governed by a Bernoulli distribution. The joint distribution over the bit vector X = (X1, . . . , Xk) is called a multivariate Bernoulli distribution.

multinomial distribution:
Every word position in an e-mail corresponds to a categorical variable with k outcomes, where k is the size of the vocabulary. The multinomial distribution manifests itself as a count vector

### 2.5 Naive Bayes Classifier
D = (x, c) = <(x<sub>1</sub>, x<sub>2</sub>,...)(features), c(class)>,
Learn P(C|X)

Continuous: Gaussian NB

Discrete: Multinomial or Multivariate

### Deep Learning

### 3.1 Perceptron algorithm
![perceptron_al](https://cdn-images-1.medium.com/max/1600/1*PbJBdf-WxR0Dd0xHvEoh4A.png)

### 3.2 Gradient Descent
![gradient_de](https://ml-cheatsheet.readthedocs.io/en/latest/_images/gradient_descent_demystified.png)


### Kernel </br>
### 1. Duality
Linearly separable data. Dual view of Perceptron + SVM
Dual Perceptron learning:</br>
![kernel_dual](https://wikimedia.org/api/rest_v1/media/math/render/svg/003c2a1743fb20ecb7bc70b1bf48bc88bde5dfbd)


### 2. Kernel Trick
How to deal with non-separable data
![svm1](https://wikimedia.org/api/rest_v1/media/math/render/svg/90e0fa283c9e642c9c11b22da45efa30b06944a9)</br>
![svm2](https://upload.wikimedia.org/wikipedia/commons/thumb/7/72/SVM_margin.png/600px-SVM_margin.png)</br>

solve: minimize |w| subject to ![svm3](https://wikimedia.org/api/rest_v1/media/math/render/svg/94c99827acb10edd809df63bb86ca1366f01a8ac) (by Lagrange multiplier)

k(x1, x2) = (1+x1·x2)<sup>2</sup>

### Ensemble </br>
### 1. Bagging</br>

Data1 ---> Model1 ----> Prediction1 (weak) </br>
Data2 ---> Model2 ----> Prediction2 (weak) </br>
...</br>
Data<sub>T</sub>----> Model<sub>T</sub> --->prediction (weak)

Vote ---> Prediction


### 2. Random Forest

Data(sample data & sample feature)

## 3. Boosting(Adaptive Boost)

weak ----> strong


Form a large set of simple features</br>
Initialize weights for training images</br>
<div>
For T rounds</br>
Normalize the weights</br>
For available features from the set, train a classifier using a single feature and evaluate the training error</br>
Choose the classifier with the lowest error</br>
Update the weights of the training images: increase if classified wrongly by this classifier, decrease if correctly</br>
Form the final strong classifier as the linear combination of the T classifiers (coefficient larger if training error is small)</br>
</div>

### unsupervised learning
K-mass clusters:

Assume k = 3</br>
1.init clusters center randomly</br>
2.Assign points X to clusters</br>
3.recompute cluster's center</br>
4.repeat 2, 3 until cluster centers don't change</br>
