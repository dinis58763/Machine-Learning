# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 15:50:40 2023

@author: djsil
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train_data = pd.read_csv('X_train.csv')
test_data = pd.read_csv('X_test.csv')

# Print the first few rows of the dataset
print(train_data.head())

# Summary statistics of the dataset
print(train_data.describe())

# Filter the data for t=0
t0_data = train_data[train_data['t'] == 0]

# Plot the positions at t=0
plt.figure(figsize=(8, 6))
plt.plot(t0_data['x_1'], t0_data['y_1'], 'o', label='Object 1')
plt.plot(t0_data['x_2'], t0_data['y_2'], 'o', label='Object 2')
plt.plot(t0_data['x_3'], t0_data['y_3'], 'o', label='Object 3')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Initial Positions of the objects at t=0')
plt.legend()
plt.grid(True)
plt.show()
