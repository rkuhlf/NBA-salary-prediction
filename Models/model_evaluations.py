import math
import numpy as np
import pandas as pd
from sklearn import metrics

from config import *

players_df = pd.read_csv(CLEANED_PLAYERS_PATH)

def get_error(model, inputs_test, outputs_test):
    predictions = model.predict(inputs_test)
    mse = metrics.mean_squared_error(outputs_test, predictions)

    rmse = np.sqrt(mse)

    return rmse

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
    players = ["LeBron James", "Steve Nash", "Dwight Howard", "Troy Murphy", "Dante Cunningham", "Stephen Curry", "Shaquille O'Neal", "Pau Gasol", "Chris Bosh", "Kobe Bryant", "Fred Roberts", "Zydrunas Ilgauskas", "James Harden", "Russell Westbrook", "Michael Jordan"]

    for player in players:
        compare_prediction(model, input_columns, player)


def compare_across_salaries(model, input_columns: list, inputs_test: pd.DataFrame, outputs_test: pd.DataFrame):
    """Graphs the error across salary levels"""

    combined = inputs_test.copy()
    combined["actual"] = outputs_test

    percentages = np.linspace(0, 1, 7)
    quantiles = []
    for percentage in percentages:
        quantiles.append(combined["actual"].quantile(percentage))
    errors = []

    prev_quantile = quantiles[0]
    for quantile in quantiles[1:]:
        selected = combined[(prev_quantile < combined["actual"]) & (combined["actual"] < quantile)]
        prev_quantile = quantile

        if len(selected) == 0:
            raise Exception("Decrease the number of bins so that everyone has one, that way we can show error over the full range.")

        predictions = model.predict(selected[input_columns])

        mse = metrics.mean_squared_error(selected["actual"], predictions)
        errors.append(math.sqrt(mse))

    return quantiles, errors
        
def predict_custom(model, custom_input: pd.DataFrame):
    return model.predict(custom_input)

def highest_prediction(model, input_columns, top=10):
    predictions = model.predict(players_df[input_columns])

    players_df["prediction"] = predictions

    return players_df.sort_values("prediction")[-top:]