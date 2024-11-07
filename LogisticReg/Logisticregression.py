import numpy as np
from matplotlib import pyplot as plt

class LogisticRegression():
    def __init__(self):
        self.weights = None
        self.bias = None
        self.cost = []
        self.min_cost = float("inf")
        
    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))
    
    def diff_sigmoid(self, sigma):
        return sigma*(1 - sigma)
    
    def fit(self, X, y, lr=0.01, epochs=10):
        m = X.shape[0]
        
        if self.weights is None:
            self.weights = np.zeros(X.shape[1])
            self.bias = 0.0
            
        for _ in range(epochs):
            fwb = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(fwb)
            loss = -np.mean(y * np.log(y_pred + 1e-15) + (1 - y) * np.log(1 - y_pred + 1e-15))
            
            self.cost.append(loss)
            
            dw = (1 / m) * np.dot(X.T, (y_pred - y))
            db = (1 / m) * np.sum(y_pred - y)
            
            self.weights -= lr * dw
            self.bias -= lr * db
            print(f'Epoch {_+1}, Loss: {loss}')
            
            if self.min_cost > loss:
                self.min_cost = loss
    
    def predict(self, X_test):
        fwb = np.dot(X_test, self.weights) + self.bias
        return self.sigmoid(fwb)
        
    
    def plot_loss(self):
        plt.plot(range(len(self.cost)), self.cost)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss vs Epoch')
        plt.show()
    
def normalize(X):
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)

