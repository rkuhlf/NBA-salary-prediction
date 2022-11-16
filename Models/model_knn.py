from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split

from Models.inputs import get_data

inputs, outputs = get_data()
inputs_train, inputs_test, outputs_train, outputs_test = train_test_split(inputs, outputs, test_size=0.3, random_state=101)

KNN_model = KNeighborsRegressor(n_neighbors=3, weights="distance", metric="minkowski")

KNN_model.fit(inputs_train, outputs_train)