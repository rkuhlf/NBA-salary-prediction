import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split

from Analysis.ModelAnalysis.model_evaluations import compare_players, compare_prediction, get_rmse
from Analysis.ModelAnalysis.prediction_analysis import predict_custom
from Data.inputs import get_inputs_train, get_outputs_train, riley_inputs_KNN, get_inputs_test, get_outputs_test


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
    model.input_columns = input_columns

    return model

def get_trained_model():
    model = create_model()
    model: KNeighborsRegressor = model.fit(get_inputs_train(input_columns), get_outputs_train())

    return model

if __name__ == "__main__":    
    model = get_trained_model()    
    

    # compare_players(model, input_columns)

    # print(predict_custom(model, riley_inputs_KNN))
    # print(get_rmse(model, get_inputs_test(model.input_columns), get_outputs_test()))

    pass