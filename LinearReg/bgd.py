import numpy as np
from matplotlib import pyplot as plt

X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

X_b = np.c_[np.ones((100, 1)), X]
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

"""
This is Batch Gradient Descent.
Large amount of computation makes it slow
eta = 0.1
n_iter = 1000
m = 100

theta = np.random.randn(2, 1)

for iteration in range(n_iter):
    gradients = 2 / m * X_b.T.dot(X_b.dot(theta) - y)
    theta = theta - eta * gradients
    
y_b = X_b.dot(theta_best)
y_pred = X_b.dot(theta)

print(theta)
print(theta_best)

plt.plot(X, y_b, "g-")
plt.plot(X, y, "r.")
plt.plot(X, y_pred, "b-")
plt.show()

"""

"""
This is stochastic gradient descent.
Use Simulated annealing in this to overcome stucking in the local minimum or a plateau

n_epochs = 50
m = 100
t0, t1 = 5, 50

def learning_schedule(t):
    return t0 / (t + t1)

theta = np.random.randn(2, 1)

for epoch in range(n_epochs):
    for i in range(m):
        random_index = np.random.randint(m)
        xi = X_b[random_index : random_index+1]
        yi = y[random_index: random_index+1]
        gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
        eta = learning_schedule(epoch*m + i)
        theta = theta - eta * gradients

y_pred = X_b.dot(theta)

# plt.plot(X, y_b, "g-")
plt.plot(X, y, "r.")
plt.plot(X, y_pred, "b-")
plt.show()
"""