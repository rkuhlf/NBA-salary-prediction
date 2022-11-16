import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from Models.inputs import get_data
from Models.model_evaluations import compare_players, get_error


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


def create_model(inputs_train, outputs_train):
    model = RandomForestRegressor(100, criterion="squared_error")

    model = model.fit(inputs_train, outputs_train)

    return model

if __name__ == "__main__":    
    inputs, outputs = get_data(input_columns)

    errors = []
    for state in [100, 101, 102]:
        inputs_train, inputs_test, outputs_train, outputs_test = train_test_split(inputs, outputs, test_size=0.3, random_state=state)
        model = create_model(inputs_train, outputs_train)
        rmse = get_error(model, inputs_test, outputs_test)
        errors.append(rmse)

        print(f"{rmse/1e6:.3f} million")

        # Error around seven million

    print(np.mean(errors))

    compare_players(model, input_columns)

    pass