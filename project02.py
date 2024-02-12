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
from sklearn.model_selection import train_test_split
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
df.drop(columns=['v_x_1', 'v_y_1', 'v_x_2', 'v_y_2', 'v_x_3', 'v_y_3'], inplace=True)

# Create a dictionary to map old column names to new column names
# Create an empty list to store the repeated DataFrames
def add_columns( data_frame):
    auxiliar = []
    for _, row in data_frame.iterrows():
        auxiliar.extend([row.to_frame().T] * 257)
    return auxiliar


repeated_rows = pd.concat(add_columns(filtered_rows), ignore_index=True)
repeated_rows = repeated_rows.drop(columns=['t'])
repeated_rows = repeated_rows.drop(columns=['Id'])
repeated_rows = pd.concat([repeated_rows, df['t']], axis=1)

original = df.copy()
repeated_rows.rename(columns={ 
    'x_1': 'x0_1',
    'y_1': 'y0_1',
    'x_2': 'x0_2',
    'y_2': 'y0_2',
    'x_3': 'x0_3',
    'y_3': 'y0_3',}, inplace=True)

original = original.drop(columns=['t'])

repetidas_mais_originais = pd.concat([repeated_rows, original], axis=1)


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

#organize the order
new_order = ['t', 'x0_1', 'y0_1', 'x0_2', 'y0_2', 'x0_3', 'y0_3','x_1','y_1','x_2','y_2','x_3','y_3']
repetidas_mais_originais = repetidas_mais_originais[new_order]

#labels identifying diferent sets
labels_of_X = ['t', 'x0_2', 'y0_2']
labels_of_y = ['x_1','y_1','x_2','y_2','x_3','y_3']

#spliting the data
train, test_split = train_test_split(repetidas_mais_originais, test_size=0.3)

#pipeline of testing
pipeline_for_testing = Pipeline([
     ('scaler', scaler),
    ('knn_regressor', knn_reg)
])

#pipeline of main
pipeline = Pipeline([
    ('scaler', scaler),
    ('knn_regressor', knn_reg)
])

#arrange test set
test_aux= test.copy()
test_aux = test_aux.drop(columns=['Id'])
test_aux = test_aux.drop(columns=['x0_1'])
test_aux = test_aux.drop(columns=['y0_1'])
test_aux = test_aux.drop(columns=['x0_3'])
test_aux = test_aux.drop(columns=['y0_3'])


#train the two models
test_split = test_split.reset_index(drop=True)
pipeline.fit(repetidas_mais_originais[labels_of_X], repetidas_mais_originais[labels_of_y])
pipeline_for_testing.fit(train[labels_of_X], train[labels_of_y])

#predict 
prediction = pipeline.predict(test_aux)
test_split_prediction = pipeline_for_testing.predict(test_split[labels_of_X])


################################### EVALUATION METRICS ####################################


test_results = test_split[labels_of_y]
total_error_sum = 0

num_cols = 0

for col in labels_of_y:
    col_error_sum = 0
    for i in range(0 , len(test_split_prediction)):
        col_error_sum += (pow(test_results[col][i]- test_split_prediction[i][num_cols], 2)) 

    
    col_mse = col_error_sum / len(test_split)

    total_error_sum += col_mse
    num_cols +=1

mse_avg = total_error_sum / num_cols
print(f'Mean Squared Error: {mse_avg}')
mean_rmse = np.sqrt(mse_avg)
print(f'Root Mean Squared Error: {mean_rmse}')

#prepare to download
prediction_data_frame = pd.DataFrame(data=prediction, columns=['x_1', 'y_1', 'x_2', 'y_2', 'x_3', 'y_3'])
prediction_data_frame_delivery = pd.concat([test["Id"], prediction_data_frame], axis=1)


################################## RESULTS AND ANALYSIS ####################################


# Get the data for id=4 from the prediction_data_frame
id_4_data = prediction_data_frame[prediction_data_frame_delivery['Id'] == 4]

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


################################## DOWNLOAD CSV FILE ##########################################


to_csv = pd.DataFrame(prediction_data_frame_delivery)
to_csv.to_csv('nova_solução.csv', index=False)


##################################### ONNX BINARY #############################################


initial_type = [('float_input', FloatTensorType([None, 3]))]
onnx_model = convert_sklearn(pipeline, initial_types=initial_type)

# Save the ONNX model to a file
with open("model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())