import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures

# Read the CSV file into a DataFrame, skipping the first row (header)
df = pd.read_csv('X_train.csv')
test = pd.read_csv('X_test.csv')

# Filter rows that are multiples of 257
filtered_rows = df.iloc[::257].copy()
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


print(len(filtered_rows))


# Create an empty list to store the repeated DataFrames
repeated_dfs = []

# Repeat each DataFrame 256 times
for _, row in filtered_rows.iterrows():
    repeated_dfs.extend([row.to_frame().T] * 257)

# Concatenate all the repeated DataFrames into a single DataFrame
repeated_rows = pd.concat(repeated_dfs, ignore_index=True)

print(len(repeated_rows))
#print(len(test))

# Create a KNeighborsRegressor model
knn_reg = KNeighborsRegressor(n_neighbors=5)

# Create a LinearRegression model
reg = LinearRegression()

# Create a PolynomialRegression model
degree = 2
poly_features = PolynomialFeatures(degree)


# Create a StandardScaler
#scaler = StandardScaler()



# Create a pipeline with scaling
pipeline = Pipeline([
    #('scaler', scaler),
   
    ('polynomial_features', poly_features),
    #('knn_regressor', knn_reg)
    ('regression', reg)
])

#eliminate the lines where there are only zeros

print(df.shape)
print(repeated_rows.shape)

'''
df = df[(df.iloc[:, :-1] != 0).any(axis=1)]

df_ids = set(df['Id'])
repeated_rows = repeated_rows[repeated_rows['Id'].isin(df_ids)]

print(df.shape)
print(repeated_rows.shape)
'''



print(df[1500:1550])



# Fit the model to your data
pipeline.fit(repeated_rows, df)
prediction= pipeline.predict(test)

#delete non usefull columns
columns_to_eliminate = [0, 3, 4, 7, 8, 11, 12, 13]
prediction = np.delete(prediction, columns_to_eliminate, axis=1)

#transform into a dataframe
prediction_data_frame = pd.DataFrame(data=prediction, columns=['x_1', 'y_1', 'x_2', 'y_2', 'x_3', 'y_3'])

#put the id column
if 'Id' in df.columns:
    prediction_data_frame['Id'] = df['Id']

#change the order
new_order = ['Id', 'x_1', 'y_1', 'x_2', 'y_2', 'x_3', 'y_3']
prediction_data_frame = prediction_data_frame[new_order]

#download csv file
to_csv = pd.DataFrame(prediction_data_frame)
to_csv.to_csv('arquivo.csv', index=False)


print(prediction_data_frame[:])
