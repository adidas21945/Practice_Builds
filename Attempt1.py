# Install necessary packages
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
import torch.nn.functional as F

# Create a Model Class
# hn refers to a hidden layer
# fc1 refers to fully connected layers
class Model(nn.Module):
    def __init__(self, in_features=4, h1=8, h2=8, out_features=3):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)

        return x

# Pick manual seed
torch.manual_seed(41)
# Create instance of model
model = Model()

url = 'https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv'
my_df = pd.read_csv(url)

# Replace species names with integers
my_df['species'] = my_df['species'].replace('setosa', 0.0)
my_df['species'] = my_df['species'].replace('versicolor', 1.0)
my_df['species'] = my_df['species'].replace('virginica', 2.0)

# Train, test, split - set X, y
X = my_df.drop('species', axis=1)
y = my_df['species']
my_df.tail()

# Convert to numpy arrays
X = X.values
y = y.values

# Train, test, split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)

# Convert to tensors
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

# Set criterion of model to measure error - how far off predictions are from data
criterion = nn.CrossEntropyLoss()

# Use Adam Optimizer
# lr refers to learning rate - if error doesn't go down, lower lr
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Train model
# Determine number of epochs - one run through all training data
epochs = 100
losses = []
for i in range(epochs):
    y_pred = model.forward(X_train) # Get predicted results
    # Measure the loss/error
    loss = criterion(y_pred, y_train) # Predicted vs y_train values
    # Keep track of losses
    losses.append(loss.detach().numpy())
    # Print every 10 epochs
    if i % 10 == 0:
        print(f'Epoch: {i} and loss: {loss}')

    # Take error rate of forward propagation and feed it back through network to fine tune weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Graph
plt.plot(range(epochs), losses)
plt.ylabel("loss/error")
plt.xlabel('Epoch')
plt.show()

# Evaluate Model on Test Data Set (validate model on test set)
with torch.no_grad():
    y_eval = model.forward(X_test) # X_test are features from our test, y_eval will be predictions
    loss = criterion(y_eval, y_test) # Find loss/error of y_eval vs y_test

correct = 0
with torch.no_grad():
    for i, data in enumerate(X_test):
        y_val = model.forward(data)

        if y_test[i] == 0:
            x = "setosa"
        elif y_test[i] == 1:
            x = "versicolor"
        elif y_test[i] == 2:
            x = "virginica"

        # Tells us what type of flower our network thinks it is
        print(f'{i+1}.) {str(y_val)} \t {x} \t {y_val.argmax().item()}')

        # Correct or not
        if y_val.argmax().item() == y_test[i]:
            correct += 1

print(f'We got {correct} correct.')

new_iris = torch.tensor([4.7, 3.2, 1.3, 0.2])
with torch.no_grad():
    print(model(new_iris))

newer_iris = torch.tensor([5.9, 3.0, 5.1, 1.8])
with torch.no_grad():
    print(model(newer_iris))

# Save NN Model
torch.save(model.state_dict(), 'iris_model.pt')

# Load the saved model
new_model = Model()
new_model.load_state_dict(torch.load('iris_model.pt'))

# Make sure it loaded correctly
new_model.eval()