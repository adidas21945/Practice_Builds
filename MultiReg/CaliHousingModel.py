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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# Load dataset
df = pd.read_csv("https://raw.githubusercontent.com/sonarsushant/California-House-Price-Prediction/refs/heads/master/housing.csv")

# Drop data given as a string
df = df.drop("ocean_proximity", axis=1)

# Find correlation
df.corr()["total_bedrooms"]

# Remove columns with weak correlation
df = df.drop(["longitude", "latitude", "housing_median_age", "median_income", "median_house_value"], axis=1)

# Remove any NaNs
df = df.dropna()

# Convert to numpy array
df_np = df.to_numpy()

# Set/scale training data
X, y = df_np[:, :3], df_np[:, -1]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate error
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(mae)
print(mse)

# Visualize model results
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual households")
plt.ylabel("Predicted households")
plt.title("Actual vs Predicted households")
plt.show()