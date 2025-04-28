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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# Import data
cancer_data = pd.read_csv("https://raw.githubusercontent.com/sharmaroshan/Breast-Cancer-Wisconsin/refs/heads/master/wisconsin.csv")

# Clean data
cancer_data.drop(["Unnamed: 32", "id"], axis = 1, inplace=True)
cancer_data.diagnosis = [1 if value == "M" else 0 for value in cancer_data.diagnosis]
cancer_data["diagnosis"] = cancer_data["diagnosis"].astype("category", copy = False)

# Divide into predictors and target
X = cancer_data.drop(["diagnosis"], axis = 1)
y = cancer_data["diagnosis"]

# Normalize data
scaler = StandardScaler() # Import scaler object
X_scaled = scaler.fit_transform(X) # Fit scaler and transform data

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.3, random_state = 42)

# Train model
log_reg = LogisticRegression() # Create Log Reg model
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test) # Predict target

# Evaluate model - ADD PLOTS FOR EVAL
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy: .2f}")
print(classification_report(y_test, y_pred))
