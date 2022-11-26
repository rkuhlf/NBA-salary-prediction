import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from Analysis.ModelAnalysis.prediction_analysis import predict_custom
from Data.inputs import get_inputs_train, get_outputs_train, riley_inputs_forest
from Analysis.ModelAnalysis.model_evaluations import compare_players, get_rmse


input_columns = ["career_g normalized",
                #  "birth_year",
                 "draft_round",
                 "draft_year",
                 "height normalized",
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
                 "attended_college",
                 "Center",
                #  "Forward",
                #  "Small Forward",
                 "Power Forward",
                #  "Guard",
                #  "Shooting Guard",
                #  "Point Guard",
                 ]


def create_model():
    model = RandomForestRegressor(100, criterion="squared_error")
    model.inputs = input_columns

    return model

def get_trained_model():
    model = create_model()
    model = model.fit(get_inputs_train(input_columns), get_outputs_train())

    return model

def get_average_error():
    errors = []
    for state in [100, 101, 102]:
        inputs_train, inputs_test, outputs_train, outputs_test = get_player_data_for_modeling(input_columns)
        model = create_model(inputs_train, outputs_train)
        rmse = get_rmse(model, inputs_test, outputs_test)
        errors.append(rmse)

        print(f"{rmse/1e6:.3f} million")

    # Error around seven million
    return np.mean(errors)

if __name__ == "__main__":    
    model = get_trained_model()
    # print(highest_prediction(model, input_columns))
    # compare_players(model, input_columns)

    print(predict_custom(model, riley_inputs_forest))

    pass