import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.model_selection import KFold, cross_val_score
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# Read the CSV file into a DataFrame, skipping the first row (header)
df = pd.read_csv('X_train.csv')
test = pd.read_csv('X_test.csv')

# Filter rows that are multiples of 257
filtered_rows = df.iloc[::257].copy()
filtered_rows_test = test.iloc[::257].copy()

# Use the .drop method to remove one or more columns
filtered_rows.drop(columns=['v_x_1', 'v_y_1', 'v_x_2', 'v_y_2', 'v_x_3', 'v_y_3'], inplace=True)

# Create a dictionary to map old column names to new column names
column_mapping = {
    'x_1': 'x0_1',
    'y_1': 'y0_1',
    'x_2': 'x0_2',
    'y_2': 'y0_2',
    'x_3': 'x0_3',
    'y_3': 'y0_3',
}


# Use the .rename method to apply the column name changes
filtered_rows.rename(columns=column_mapping, inplace=True)

# Define the new column order and column names
new_order = ['Id', 't', 'x0_1', 'y0_1', 'x0_2', 'y0_2', 'x0_3', 'y0_3']

# Reorder the columns
filtered_rows = filtered_rows[new_order]


# Create an empty list to store the repeated DataFrames
def add_columns( data_frame):
    auxiliar = []
    for _, row in data_frame.iterrows():
        auxiliar.extend([row.to_frame().T] * 257)
    return auxiliar

test_extended = pd.concat(add_columns(filtered_rows_test), ignore_index=True)
test_extended = test_extended.drop(columns=['t'])
test_extended = pd.concat([test_extended, test['t']], axis=1)
   

repeated_rows = pd.concat(add_columns(filtered_rows), ignore_index=True)
repeated_rows = repeated_rows.drop(columns=['t'])
repeated_rows = pd.concat([repeated_rows, df['t']], axis=1)

print(len(repeated_rows))
print(repeated_rows)
print(df)


#print(len(test))

# Create a KNeighborsRegressor model
knn_reg = KNeighborsRegressor(n_neighbors=5)

# Create a LinearRegression model
reg = LinearRegression()

# Create a PolynomialRegression model
degree = 2
poly_features = PolynomialFeatures(degree)


# Create a StandardScaler
scaler = StandardScaler()
# You can adjust the alpha parameter as needed

# Create a Ridge model
ridge_reg = Ridge(alpha=1.0)  

# Wrap the Ridge estimator with MultiOutputRegressor
multioutput_regressor = MultiOutputRegressor(ridge_reg)
voting_reg = VotingRegressor([('knn', knn_reg), ('linear', reg), ('ridge', multioutput_regressor)])

# Create a pipeline with scaling
pipeline = Pipeline([
    ('scaler', scaler),
    ('polynomial_features', poly_features),
    ('regression', multioutput_regressor)
])

#('scaler', scaler),
#('knn_regressor', knn_reg)

#eliminate the lines where there are only zeros

#resolve some id inconsistence
repeated_rows = repeated_rows.drop(columns=['Id'])
repeated_rows = pd.concat([repeated_rows, df['Id']], axis=1)

test_extended = test_extended.drop(columns=['Id'])
test_extended = pd.concat([test_extended, test['Id']], axis=1)


# orgnaize the ids
new_order = ['Id', 't', 'x0_1', 'y0_1', 'x0_2', 'y0_2', 'x0_3', 'y0_3']
test_extended = test_extended[new_order]
repeated_rows = repeated_rows[new_order]


# eliminate the zeros
df = df[(df.iloc[:, :-1] != 0).any(axis=1)]

unique_ids_df = set(df['Id'])

repeated_rows = repeated_rows[repeated_rows['Id'].isin(unique_ids_df)]

# put a column identifying each group of point
repeated_rows['id_of_group'] = 0
repeated_rows['id_of_group'] = (repeated_rows['t'] == 0).cumsum()

#put a column identifying each group of point on test
test_extended['id_of_group'] = 0
test_extended['id_of_group'] = (test_extended['t'] == 0).cumsum()

# Fit the model to your data

df = df.drop(columns=['t'])
df = df.drop(columns=['v_x_1'])
df = df.drop(columns=['v_y_1'])
df = df.drop(columns=['v_x_2'])
df = df.drop(columns=['v_y_2'])
df = df.drop(columns=['v_x_3'])
df = df.drop(columns=['v_y_3'])
df = df.drop(columns=['Id'])


pipeline.fit(repeated_rows, df)
prediction = pipeline.predict(test_extended)

#transform into a dataframe
prediction_data_frame = pd.DataFrame(data=prediction, columns=['x_1', 'y_1', 'x_2', 'y_2', 'x_3', 'y_3'])

#put the id column
prediction_data_frame['Id'] = test['Id']

#change the order
new_order = ['Id', 'x_1', 'y_1', 'x_2', 'y_2', 'x_3', 'y_3']
prediction_data_frame = prediction_data_frame[new_order]

#download csv file
to_csv = pd.DataFrame(prediction_data_frame)
to_csv.to_csv('arquivo_optimized.csv', index=False)


################################### EVALUATION METRICS ####################################


kf = KFold(n_splits=5, shuffle=True, random_state=42) 
scores = cross_val_score(pipeline,repeated_rows, df, cv=kf, scoring='neg_mean_squared_error')
mean_mse = -scores.mean()
mean_rmse = np.sqrt(mean_mse)
print(f'Root Mean Squared Error: {mean_rmse}')
print(f'Mean Squared Error: {mean_mse}')


################################## RESULTS AND ANALYSIS ####################################


# Get the data for id=4 from the prediction_data_frame
id_4_data = prediction_data_frame[prediction_data_frame['Id'] == 4]

# Extract the positions for each body
body1_x_positions = id_4_data['x_1']
body1_y_positions = id_4_data['y_1']
body2_x_positions = id_4_data['x_2']
body2_y_positions = id_4_data['y_2']
body3_x_positions = id_4_data['x_3']
body3_y_positions = id_4_data['y_3']

# Generate 2D trajectory plots for each celestial body with larger markers
plt.figure(figsize=(10, 6))
plt.plot(body1_x_positions, body1_y_positions, 'o-', label='Body 1', markersize=10)
plt.plot(body2_x_positions, body2_y_positions, 'o-', label='Body 2', markersize=10)
plt.plot(body3_x_positions, body3_y_positions, 'o-', label='Body 3', markersize=10)
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('Position Plot of Celestial Bodies for Id = 4')
plt.legend()
plt.show()


##################################### ONNX BINARY ##########################################


initial_type = [('float_input', FloatTensorType([None, 9]))]
onnx_model = convert_sklearn(pipeline, initial_types=initial_type)

# Save the ONNX model to a file
with open("model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())