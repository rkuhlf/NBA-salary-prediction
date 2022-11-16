import numpy as np
import pandas as pd

from sklearn import metrics


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

def compare_players(model, input_columns):
    players = ["LeBron James", "Steve Nash", "Dwight Howard", "Troy Murphy", "Dante Cunningham", "Stephen Curry", "Shaquille O'Neal", "Pau Gasol", "Chris Bosh", "Kobe Bryant", "Fred Roberts", "Zydrunas Ilgauskas"]

    for player in players:
        compare_prediction(model, input_columns, player)
