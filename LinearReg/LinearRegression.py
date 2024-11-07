import numpy as np
from matplotlib import pyplot as plt

class LinearRegression():
    def __init__(self):
        self.weights = None
        self.bias = None
        self.cost = []
        self.min_cost = float("inf")
        
    def fit(self, X, y, lr=0.01, epochs=10):
        m, n = X.shape
        
        if self.weights is None:
            self.weights = np.random.randn(n, 1)
            self.bias    = 0.0
            
        for _ in range(epochs):
            y_pred = np.dot(X, self.weights) + self.bias
            loss   = np.mean((y_pred - y) ** 2) * 0.5
            
            self.cost.append(loss)
            
            dw = (2 / m) * np.dot(X.T, (y_pred - y))
            db = (2 / m) * np.sum(y_pred - y)
            
            self.weights -= lr * dw
            self.bias    -= lr * db
            
            if self.min_cost > loss:
                self.min_cost = loss
        
    def predict(self, X_test):
        return np.dot(X_test, self.weights) + self.bias    
    
    def plot_loss(self):
        plt.plot( self.cost, range(len(self.cost)))
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss vs Epoch')
        plt.show()
        
    def get_weights(self):
        return self.weights.reshape(1), self.bias
        
def normalize(X):
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)