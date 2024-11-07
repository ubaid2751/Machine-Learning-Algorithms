import torch
import torch.nn as nn
import torch.nn.functional as F 
import matplotlib.pyplot as plt
import struct
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import train_test_split
import numpy as np
import random

def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)
    
def load_mnist_data(imagesfile, labelsfile):
    images = read_idx(imagesfile)
    labels = read_idx(labelsfile)
    return images, labels

def visualize_images(images, labels):
    fig, axes = plt.subplots(5, 4, figsize=(10, 5))
    axes = axes.flatten()
    
    for i in range(20):
        idx = torch.randint(0, images.shape[0] - 1, size=(1,)).item()
        axes[i].imshow(images[idx])
        axes[i].set_title(labels[idx])
        axes[i].axis('off')
        
    plt.tight_layout()
    plt.show()
    
def flatten_image(images, labels):
    return images.reshape(images.shape[0], 28, 28), labels

class neuralNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(neuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, 64)
        self.layer4 = nn.Linear(64, output_size)
            
    def forward(self, xin):
        x = F.relu(self.layer1(xin))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return self.layer4(x)
    
    def predict(self, x_test):
        with torch.no_grad():
            logits = self.forward(x_test)
            return torch.argmax(logits, dim=1)
    
    def estimate_loss(self, X_val, y_val):
        self.eval()
        with torch.no_grad():
            logits = self.forward(X_val)
            loss = F.cross_entropy(logits, y_val)
        self.train()
        return loss.item()
    
    def train_model(self, X_train, y_train, epochs=100, X_val=None, y_val=None):
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.01)
        
        for epoch in range(epochs):
            self.train()
            logits = self.forward(X_train)
            loss = F.cross_entropy(logits, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            val_loss = self.estimate_loss(X_val, y_val) if X_val is not None and y_val is not None else None
            print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}, Val_Loss: {val_loss:.4f}')

train_images_path = "data\\train-images.idx3-ubyte"
train_labels_path = "data\\train-labels.idx1-ubyte"
test_images_path= "data\\t10k-images.idx3-ubyte"
test_labels_path= "data\\t10k-labels.idx1-ubyte"

train_images, train_labels = load_mnist_data(train_images_path, train_labels_path)
test_images, test_labels = load_mnist_data(test_images_path, test_labels_path)

Xtr, ytr = flatten_image(train_images, train_labels)
Xte, yte = flatten_image(test_images, test_labels)

X = torch.tensor(Xtr, dtype=torch.float32).reshape(-1 ,784) / 255.0
y = torch.tensor(ytr, dtype=torch.long)
X_test = torch.tensor(Xte, dtype=torch.float32).reshape(-1, 784) / 255.0
y_test = torch.tensor(yte, dtype=torch.long)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2,random_state=121) 

model = neuralNetwork(784, 10)
print("Training started: ")
model.train_model(X_train, y_train, epochs=100, X_val=X_val, y_val=y_val)
ypred = np.array(model.predict(X_test))

visualize_images(Xte, ypred)