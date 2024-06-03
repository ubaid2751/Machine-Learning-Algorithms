import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from Logisticregression import *

def plot_heatmap(df):
    corr = df.corr()
    plt.figure(figsize=(10, 10))
    sns.heatmap(corr, cmap='Blues', annot=True)
    plt.show()

def plot_histograms(df):
    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(10, 10))
    axes = axes.flatten()
    
    for i, col in enumerate(df.columns):
        sns.histplot(df[col], bins=50, color='g', ax=axes[i]);
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Frequency')
        
    plt.tight_layout()
    plt.show()

df = pd.read_csv(r"E:\Implementation Of Algos\data\diabetes_data.csv");

print(df.head(5));
print(df.info());
print(df.shape);

# plot_heatmap(df);
y_train = df['Outcome']
X_train = df.drop(columns=['Outcome'], axis=1);
# plot_histograms(X_train);

y_train = normalize(y_train)
X_train = normalize(X_train)

model = LogisticRegression()
model.fit(X_train, y_train, lr=0.01, epochs=100)
model.plot_loss()