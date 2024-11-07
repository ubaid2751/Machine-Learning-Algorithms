import numpy as np
from LinearRegression import *
from matplotlib import pyplot as plt

X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

X_b = np.c_[np.ones((100, 1)), X]
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

# print(theta_best)

X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new]
y_pred = X_new_b.dot(theta_best)
# print(y_pred)

rate = [10, 3, 1, 1e-1, 1e-2, 1e-3]
good_rate = 0
minCost = float("inf") 

for r in rate:
    model = LinearRegression()
    model.fit(X, y, lr=r, epochs=75)
    # print(f"Min cost is {model.min_cost}")
    
    if model.min_cost < minCost:
        good_rate = r
        minCost = model.min_cost
        w, b = model.get_weights()

print(f"Best rate will be: {good_rate}, With min cost of {minCost}")

y_pred_model = X.dot(w) + b
print(w)
print(b)

plt.plot(X_new, y_pred, "r-")
plt.plot(X, y, "b.")
plt.plot(X, y_pred_model, "g-")
plt.axis([0, 2, 0, 15])
plt.show()