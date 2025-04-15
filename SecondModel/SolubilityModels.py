# Import necessary libraries
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import torch.nn.functional as F

# Load dataset
df = pd.read_csv("https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master/delaney_solubility_with_descriptors.csv")

# Split dataset into features and target
y = df["logS"]
X = df.drop("logS", axis=1)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

# Build model
# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)

# Apply model to make a prediction
y_lr_train_pred = lr.predict(X_train)
y_lr_test_pred = lr.predict(X_test)

# View results
print(y_lr_train_pred, y_lr_test_pred)

# Evaluate model performance
lr_train_mse = mean_squared_error(y_train, y_lr_train_pred)
lr_train_r2 = r2_score(y_train, y_lr_train_pred)
lr_test_mse = mean_squared_error(y_test, y_lr_test_pred)
lr_test_r2 = r2_score(y_test, y_lr_test_pred)

# View results
print("LR MSE train: ", lr_train_mse)
print("LR R2 train: ", lr_train_r2)
print("LR MSE test: ", lr_test_mse)
print("LR R2 test: ", lr_test_r2)

lr_results = pd.DataFrame(["Linear Regression", lr_train_mse, lr_train_r2, lr_test_mse, lr_test_r2]).transpose()
lr_results.columns = ["Model", "Train MSE", "Train R2", "Test MSE", "Test R2"]

# View table of results
print(lr_results)

# Random Forest
# Train model
rf = RandomForestRegressor(max_depth=2, random_state=100)
rf.fit(X_train, y_train)

# Apply model to make a prediction
y_rf_train_pred = rf.predict(X_train)
y_rf_test_pred = rf.predict(X_test)

# Evaluate model performance
rf_train_mse = mean_squared_error(y_train, y_rf_train_pred)
rf_train_r2 = r2_score(y_train, y_rf_train_pred)
rf_test_mse = mean_squared_error(y_test, y_rf_test_pred)
rf_test_r2 = r2_score(y_test, y_rf_test_pred)

rf_results = pd.DataFrame(["Random Forest", rf_train_mse, rf_train_r2, rf_test_mse, rf_test_r2]).transpose()
rf_results.columns = ["Model", "Train MSE", "Train R2", "Test MSE", "Test R2"]

# View table of results
print(rf_results)

# Compare models
df_models = pd.concat([lr_results, rf_results], axis=0)
df_models.reset_index(drop=True)

# View table of comparison
print(df_models)

# Data viz of prediction results
plt.figure(figsize=(5,5))
plt.scatter(x = y_train, y = y_lr_train_pred, alpha = 0.3)

x = np.polyfit(y_train, y_lr_train_pred, 1)
p = np.poly1d(x)

plt.plot(y_train, p(y_train), "#F8766D")
plt.ylabel("Predicted LogS")
plt.xlabel("Experimental LogS")
plt.show()