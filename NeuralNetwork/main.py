import numpy as np

class NeuralNetwork:
    def __init__(self, layer_size, learning_rate=0.1):
        self.layer_sizes = layer_size
        self.learning_rate = learning_rate
        self.parameters = self.initialize_params()
        
    def initialize_params(self):
        np.random.seed(0)
        parameters = {}
        L = len(self.layer_sizes)
        
        for l in range(1, L):
            parameters[f'W{l}'] = np.random.rand(self.layer_sizes[l], self.layer_sizes[l-1]) * 0.01
            parameters[f'b{l}'] = np.zeros((self.layer_sizes[l], 1))
            
        return parameters
    
    def relu(self, Z):
        return np.maximum(0, Z)
    
    def relu_derivative(self, Z):
        return np.where(Z > 0, 1, 0)
    
    def linear(self, Z):
        return Z
        
    def forward(self, X):
        cache = {'A0': X}
        A = X
        L = len(self.layer_sizes) - 1
        
        for l in range(1, L+1):
            W = self.parameters[f'W{l}']
            b = self.parameters[f'b{l}']
            Z = np.dot(W, A) + b 
            A = self.linear(Z)
            cache[f'Z{l}'] = Z
            cache[f'A{l}'] = A
        
        return A, cache
    
    def backward_prop(self, cache, X, Y):
        gradients = {}
        L = len(self.layer_sizes) - 1
        m = X.shape[1]
        
        dAl = cache[f'A{L}'] - Y
        
        for l in reversed(range(1, L+1)):
            dW = 1 / m * np.dot(dAl, cache[f'A{l-1}'].T)
            dB = 1 / m * np.sum(dAl, axis=1, keepdims=True)
            gradients[f'dW{l}'] = dW
            gradients[f'db{l}'] = dB
            
            if l > 1:
                dA_prev = np.dot(self.parameters[f'W{l}'].T, dAl)
                dAl = dA_prev * self.relu_derivative(cache[f'Z{l-1}'])
                
        return gradients
    
    def update_parameters(self, gradients):
        L = len(self.layer_sizes) - 1
        
        for l in range(1, L+1):
            self.parameters[f'W{l}'] -= self.learning_rate * gradients[f'dW{l}']
            self.parameters[f'b{l}'] -= self.learning_rate * gradients[f'db{l}']
            
    def compute_cost(self, AL, Y):
        m = Y.shape[1]
        cost = (1/(2*m)) * np.sum(np.square(AL - Y))
        cost = np.squeeze(cost)
        return cost
            
    def train(self, X, Y, epochs):
        for epoch in range(epochs):
            AL, cache = self.forward(X)
            cost = self.compute_cost(AL, Y)
            gradients = self.backward_prop(cache, X, Y)
            self.update_parameters(gradients)
            
            if epoch % 20 == 0:
                print(f'Epoch {epoch+1}, Cost: {cost:.3f}')
                
    def predict(self, X):
        Al, _ = self.forward(X)
        return Al
    
np.random.seed(42)
X = np.random.randn(3, 500)  # 3 features, 500 examples
y = 3 * X[0] + 2 * X[1] + X[2] + np.random.randn(500)  # Linear combination with noise

# Reshape y to match the dimensions (1, number of examples)
Y = y.reshape(1, -1)

# Define neural network with multiple layers for regression
layer_sizes = [3, 5, 5, 1]  # Input layer, 2 hidden layers, output layer
nn = NeuralNetwork(layer_sizes, learning_rate=0.01)

# Train the neural network
nn.train(X, Y, epochs=10000)

# Test the trained neural network
print("\nFinal output after training:")
predictions = nn.predict(X)
print(predictions[:, :10])