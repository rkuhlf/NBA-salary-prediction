import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from Models.inputs import get_data

from sklearn import metrics

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


def get_error(model, inputs_test, outputs_test):
    predictions = model.predict(inputs_test)
    mse = metrics.mean_squared_error(outputs_test, predictions)

    rmse = np.sqrt(mse)

    return rmse


players_df = pd.read_csv("./cleaned_players.csv")

def compare_prediction(model, input_columns, name):
    player_data = players_df[players_df["name"] == name]

    if len(player_data) > 1:
        raise Exception("Multiple player names")
    if len(player_data) == 0:
        raise Exception("Could not find player")

    predicted_salary = model.predict(player_data[input_columns])

    print(f"\n---{name}---")
    print(f"Predicted is {predicted_salary[0] / 1e6:.2f} million")

    actual = player_data["adjusted_career_revenue"].values[0] / 1e6
    print(f"Actual is {actual:.2f} million")

def compare_players(model):
    players = ["LeBron James", "Steve Nash", "Dwight Howard", "Troy Murphy", "Dante Cunningham", "Stephen Curry", "Shaquille O'Neal", "Pau Gasol", "Chris Bosh", "Kobe Bryant", "Fred Roberts", "Zydrunas Ilgauskas"]

    for player in players:
        compare_prediction(model, input_columns, player)


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

    compare_players(model)

    pass