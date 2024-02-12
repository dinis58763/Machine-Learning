# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 15:27:07 2023

@author: djsil
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.compose import ColumnTransformer
import time


train_data = pd.read_csv('X_train.csv')
test_data = pd.read_csv('X_test.csv')
print(train_data.head())

train_data.columns

idx = np.hstack((0, train_data[train_data.t == 10].index.values + 1))
idx.shape, train_data.t.min(), train_data.t.max()

k = np.random.randint(idx.shape[0])
print(k)
print(idx[k])
pltidx = range (idx[k], 257 + idx[k])
pltsquare = idx[k]

plt.plot(train_data.x_1[pltidx], train_data.y_1[pltidx])
plt.plot(train_data.x_2[pltidx], train_data.y_2[pltidx])
plt.plot(train_data.x_3[pltidx], train_data.y_3[pltidx])

plt.plot(train_data.x_1[pltsquare], train_data.y_1[pltsquare], 's')
plt.plot(train_data.x_2[pltsquare], train_data.y_2[pltsquare], 's')
plt.plot(train_data.x_3[pltsquare], train_data.y_3[pltsquare], 's')


# Randomly select a time step t within the range of training data
t_random = np.random.uniform(train_data.t.min(), train_data.t.max())

# Find the closest time step in the training data
closest_time_step = train_data['t'].sub(t_random).abs().idxmin()

# Extract the positions and velocities at the closest time step
x_1 = train_data.loc[closest_time_step, 'x_1']
y_1 = train_data.loc[closest_time_step, 'y_1']
v_x_1 = train_data.loc[closest_time_step, 'v_x_1']
v_y_1 = train_data.loc[closest_time_step, 'v_y_1']

x_2 = train_data.loc[closest_time_step, 'x_2']
y_2 = train_data.loc[closest_time_step, 'y_2']
v_x_2 = train_data.loc[closest_time_step, 'v_x_2']
v_y_2 = train_data.loc[closest_time_step, 'v_y_2']

x_3 = train_data.loc[closest_time_step, 'x_3']
y_3 = train_data.loc[closest_time_step, 'y_3']
v_x_3 = train_data.loc[closest_time_step, 'v_x_3']
v_y_3 = train_data.loc[closest_time_step, 'v_y_3']

print("t = " + str(t_random) + " : " + " : " + str(x_1) + " : " + str(y_1) +" : "+ str(v_x_1) + " : " + str(v_y_1))

# Visualise the positions and velocities at the selected time step
plt.figure(figsize=(8, 6))
plt.plot(x_1, y_1, 'o', label='Object 1', markersize=10)
plt.quiver(x_1, y_1, v_x_1, v_y_1, angles='xy', scale_units='xy', scale=1, color='blue')

plt.plot(x_2, y_2, 'o', label='Object 2', markersize=10)
plt.quiver(x_2, y_2, v_x_2, v_y_2, angles='xy', scale_units='xy', scale=1, color='green')

plt.plot(x_3, y_3, 'o', label='Object 3', markersize=10)
plt.quiver(x_3, y_3, v_x_3, v_y_3, angles='xy', scale_units='xy', scale=1, color='red')

plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title(f'Trajectory of Objects at t = {t_random:.2f}')
plt.legend()
plt.grid(True)
plt.show()


# Prepare the training data
X_train = train_data[['x_1', 'y_1', 'x_2', 'y_2', 'x_3', 'y_3']]
y_train = train_data[['v_x_1', 'v_y_1', 'v_x_2', 'v_y_2', 'v_x_3', 'v_y_3']]

# Initialize a Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions for the test data
X_test = test_data[['x0_1', 'y0_1', 'x0_2', 'y0_2', 'x0_3', 'y0_3']]

# Rename the columns to match the training data feature names
X_test.columns = ['x_1', 'y_1', 'x_2', 'y_2', 'x_3', 'y_3']

y_pred = model.predict(X_test)
print(y_pred)


# Extract the true target values for comparison
true_targets = test_data[['x0_1', 'y0_1', 'x0_2', 'y0_2', 'x0_3', 'y0_3']]


# Step 5: Evaluation - model's performance
mse = mean_squared_error(true_targets, y_pred)
print(f"Mean Squared Error: {mse}")


# Step 5: Predict Positions and Velocities on Test Data

# Use the trained model to predict positions and velocities on the test data
X_test = test_data[['x0_1', 'y0_1', 'x0_2', 'y0_2', 'x0_3', 'y0_3']]
# Rename the columns to match the training data feature names
X_test.columns = ['x_1', 'y_1', 'x_2', 'y_2', 'x_3', 'y_3']

test_predictions = model.predict(X_test)

# Separate the predicted positions and velocities
predicted_positions = test_predictions[:, :6]

# Step 6: Visualize Predictions (Optional)

# Example: Visualize the predicted trajectory of object 1
plt.figure(figsize=(8, 6))
plt.plot(predicted_positions[:, 0], predicted_positions[:, 1], label='Predicted Object 1')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Predicted Trajectory of Object 1')
plt.legend()
plt.grid(True)
plt.show()









# Generate the submission file
# submission = test_data[['Id']]
# submission[['x_1', 'y_1', 'x_2', 'y_2', 'x_3', 'y_3']] = y_test_predictions

# Save the submission file
# submission.to_csv('submission.csv', index=False)

