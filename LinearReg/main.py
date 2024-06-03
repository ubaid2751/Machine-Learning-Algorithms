import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from LinearRegression import *
import seaborn as sns

def plot_heatmap():
    correlation = df.corr()
    plt.figure(figsize=(10,10))
    sns.heatmap(data=correlation, annot=True, cmap='Blues')
    plt.show()

def plot_histograms():
    fig, axes = plt.subplots(nrows=7, ncols=2, figsize=(10, 10))
    axes = axes.flatten()

    for i, col in enumerate(df.columns):
        sns.histplot(df[col], bins=50, color='g', ax=axes[i])
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()
    
def plot_cost_vs_epochs(cost, epoch):
    plt.plot(epoch, cost)
    plt.xlabel('Epochs')
    plt.ylabel('Cost')
    plt.show()

df = pd.read_csv("E:\Implementation Of Algos\data\data.csv")

df[['state', 'zip']] = df['statezip'].str.split(' ', expand=True)
df = df.dropna(axis=1)

columns_to_drop = ['street', 'date', 'country', 'city', 'statezip', 'state']
df = df.drop(columns=columns_to_drop, axis=1)

df['zip'] = df['zip'].astype(int)

y_train = df['price']
X_train = df.drop(columns=['price'], axis=1)

y_train = normalize(y_train)
X_train = normalize(X_train)

model = LinearRegression()

epoch = np.array([10, 20, 30, 40, 50, 70, 90, 120])
cost = np.zeros(shape=epoch.size)

for i, ep in enumerate(epoch):
    model.__fit__(X=X_train, y=y_train, lr=0.01, epochs=ep)
    cost[i] = model.min_cost
    
plot_cost_vs_epochs(cost, epoch)