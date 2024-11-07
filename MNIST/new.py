import torch
import torch.nn as nn
import torch.nn.functional as F
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

import warnings
warnings.filterwarnings('ignore') 

df_train = pd.read_csv("data\\train.csv")
df_test = pd.read_csv("data\\test.csv")

for col in df_train.columns:
    print(f"{col} have: {df_train[col].nunique()} unique values")
    
def handle_missing_values(data: pd.DataFrame, threshold: int = 10):
        numeric_list = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        for col in data.columns:
            if data[col].dtype == 'object' and data[col].nunique() <= threshold:
                data[col].fillna(data[col].mode()[0], inplace=True)
            elif data[col].dtype in numeric_list:
                data[col].fillna(data[col].median(), inplace=True)

def encoding_features(data: pd.DataFrame, encoding_threshold:int=5):
        label_encoder = LabelEncoder()
        
        object_columns = data.select_dtypes(include=['object']).columns
        for col in object_columns:
            if data[col].nunique() <= encoding_threshold:
                data[col] = data[col].astype('category')
                data[col] = label_encoder.fit_transform(data[col])

def drop_object_features(data: pd.DataFrame, category_threshold: int=5):
    object_columns = data.select_dtypes(include=['object']).columns
    cols_to_drop = []
    for col in object_columns:
        if data[col].nunique() > category_threshold:
            cols_to_drop.append(col)
            
    data.drop(cols_to_drop, axis=1, inplace=True)

df_train.drop('Cabin', axis=1, inplace=True)
df_train.drop('PassengerId', axis=1, inplace=True)

handle_missing_values(data=df_train, threshold=200)
encoding_features(df_train, encoding_threshold=5)
drop_object_features(df_train)

handle_missing_values(df_test, threshold=200)
encoding_features(df_test, encoding_threshold=5)
drop_object_features(df_test)

class NN(nn.Module):
    def __init__(self, input_size, output_size):
        super(NN, self).__init__()
        self.layer1 = nn.Linear(input_size, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, 64)
        self.layer4 = nn.Linear(64, output_size)
    
    def forward(self, xin):
        x = F.tanh(self.layer1(xin))
        x = F.tanh(self.layer2(x))
        x = F.tanh(self.layer3(x))
        x = self.layer4(x)
        return F.log_softmax(x, dim = 1)
    
    def predict(self, X):
        with torch.no_grad():
            logits = self.forward(X)
            return torch.argmax(logits, dim=1)
    
    def eval_loss(self, X, y):
        self.eval()
        with torch.no_grad():
            logits = self.forward(X)
            loss = F.nll_loss(logits, y)
        self.train()
        return loss.item()
    
    def train_model(self, X, y, epochs=100, X_val=None, y_val=None):
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.001)
        
        for epoch in range(epochs):
            self.train()
            logits = self.forward(X)
            loss = F.nll_loss(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
#             val_loss = self.estimate_loss(X_val, y_val) if X_val is not None and y_val is not None else None
            print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}, Val_Loss: {None}')
        
X = df_train.drop(columns=['Survived']).values.astype(dtype=np.float32)
Y = df_train['Survived'].values.astype(dtype=np.int64)

X_train = torch.tensor(X)
y_train = torch.tensor(Y)

model = NN(X_train.shape[1], output_size=2)
model.train_model(X_train, y_train, epochs=100)

df_test = df_test.drop(columns=['PassengerId'], axis=1)

X_te = df_test.values.astype(dtype=np.float32)
X_test = torch.tensor(X_te)

ypred = np.array(model.predict(X_test))




















