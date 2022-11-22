import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split

from Analysis.ModelAnalysis.model_evaluations import compare_players, compare_prediction, get_rmse
from Data.inputs import get_inputs_train, get_outputs_train


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


def create_model():
    model = KNeighborsRegressor(n_neighbors=4, weights="uniform", metric="minkowski")
    # Getting about eight million in root-mean-squared-error
    model.inputs = input_columns

    return model

def get_trained_model():
    model = create_model()
    model: KNeighborsRegressor = model.fit(get_inputs_train(input_columns), get_outputs_train())

    return model

def get_average_error(model_func, inputs, outputs):
    errors = []
    for state in [100, 101, 102]:
        inputs_train, inputs_test, outputs_train, outputs_test = train_test_split(inputs, outputs, test_size=0.3, random_state=state)
        model = model_func(inputs_train, outputs_train)
        rmse = get_rmse(model, inputs_test, outputs_test)
        errors.append(rmse)

        print(f"{rmse/1e6:.3f} million")

    return np.mean(errors)

if __name__ == "__main__":    
    model = get_trained_model()    
    

    # compare_players(model, input_columns)

    # TODO: predict on myself
    inputs = {
            "career_g normalized": 0,
            "draft_year": 0,
            "weight normalized": 0,
            "career_pts normalized": 0,
            "career_per normalized": 0,
            "career_ws normalized": 0,
            "career_fg3 normalized": 0,
            "Forward": 0,
            "Small Forward": 0,
            "Shooting Guard": 0,
    }
    # print(predict_custom(model, ))

    pass