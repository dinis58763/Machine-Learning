# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 15:08:34 2023

@author: djsil
"""

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Step 1: Load the Data
train_data = pd.read_csv('X_train.csv')
test_data = pd.read_csv('X_test.csv')

# Step 2: Data Preprocessing

# 2.1: Split the data into features (X) and labels (y)
X = test_data[['x0_1', 'y0_1', 'x0_2', 'y0_2', 'x0_3', 'y0_3']]
y = train_data[['x_1', 'y_1', 'x_2', 'y_2', 'x_3', 'y_3', 'v_x_1', 'v_y_1', 'v_x_2', 'v_y_2', 'v_x_3', 'v_y_3']]

print("Number of samples in X:", X.shape[0])
print("Number of samples in y:", y.shape[0])

# 2.2: Normalize the data (optional, depending on the model)

# Step 3: Model Selection and Training

# 3.1: Split the data into training and validation sets (e.g., 80% training, 20% validation)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 3.2: Initialize and train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 4: Model Evaluation and Comparison

# 4.1: Make predictions on the validation set
y_pred = model.predict(X_val)

# 4.2: Evaluate the model's performance (e.g., using Mean Squared Error)
mse = mean_squared_error(y_val, y_pred)
print("Mean Squared Error:", mse)

# Step 5: Predict Positions and Velocities on Test Data

# Use the trained model to predict positions and velocities on the test data
X_test = test_data[['x0_1', 'y0_1', 'x0_2', 'y0_2', 'x0_3', 'y0_3']]
test_predictions = model.predict(X_test)

# Separate the predicted positions and velocities
predicted_positions = test_predictions[:, :6]
predicted_velocities = test_predictions[:, 6:]

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

# Step 7: Save Predictions to a CSV file (for submission)
# You should format the predictions according to the submission requirements.

# For example, you can create a DataFrame for the predictions and save it to a CSV file:
submission_df = pd.DataFrame({'x_1': predicted_positions[:, 0], 'y_1': predicted_positions[:, 1],
                              'x_2': predicted_positions[:, 2], 'y_2': predicted_positions[:, 3],
                              'x_3': predicted_positions[:, 4], 'y_3': predicted_positions[:, 5],
                              'v_x_1': predicted_velocities[:, 0], 'v_y_1': predicted_velocities[:, 1],
                              'v_x_2': predicted_velocities[:, 2], 'v_y_2': predicted_velocities[:, 3],
                              'v_x_3': predicted_velocities[:, 4], 'v_y_3': predicted_velocities[:, 5]})

# Save the DataFrame to a CSV file
submission_df.to_csv('predictions.csv', index=False)
