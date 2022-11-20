import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split

from Models.inputs import get_data
from Models.model_evaluations import compare_players, compare_prediction, get_error, compare_across_salaries


input_columns = ["career_g normalized",
                #  "birth_year",
                #  "draft_round",
                 "draft_year",
                #  "height normalized",
                 "weight normalized",
                 "career_pts normalized",
                 "career_per normalized",
                 "career_ws normalized",
                #  "career_ast normalized",
                #  'career_fg normalized',
                 "career_fg3 normalized",
                #  "career_trb normalized",
                #  'career_ft normalized',
                #  "birth_month",
                #  "attended_college",
                #  "Center",
                 "Forward",
                 "Small Forward",
                #  "Power Forward",
                #  "Guard",
                 "Shooting Guard",
                #  "Point Guard",
                 ]


def create_model(inputs_train, outputs_train):
    KNN_model = KNeighborsRegressor(n_neighbors=4, weights="uniform", metric="minkowski")

    KNN_model = KNN_model.fit(inputs_train, outputs_train)

    return KNN_model

def get_trained_model():
    inputs, outputs = get_data(input_columns)
    # print(get_average_error(create_model))

    inputs_train, inputs_test, outputs_train, outputs_test = train_test_split(inputs, outputs, test_size=0.3, random_state=101)
    model = create_model(inputs_train, outputs_train)

    return model, inputs_test, outputs_test

def get_average_error(model_func, inputs, outputs):
    errors = []
    for state in [100, 101, 102]:
        inputs_train, inputs_test, outputs_train, outputs_test = train_test_split(inputs, outputs, test_size=0.3, random_state=state)
        model = model_func(inputs_train, outputs_train)
        rmse = get_error(model, inputs_test, outputs_test)
        errors.append(rmse)

        print(f"{rmse/1e6:.3f} million")

    return np.mean(errors)

if __name__ == "__main__":    
    model, inputs_test, outputs_test = get_trained_model()
    quantiles, errors = compare_across_salaries(model, input_columns, inputs_test, outputs_test)
    
    # Getting about eight million in root-mean-squared-error

    compare_players(model, input_columns)

    pass