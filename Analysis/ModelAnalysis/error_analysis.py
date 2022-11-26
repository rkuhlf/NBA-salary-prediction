"""Functions visualizing which parts of the data set have the most error and comparing the errors across different models."""

import math
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import sklearn

from Data.inputs import get_inputs_test, get_outputs_test, get_player_data, positions
from Analysis.ModelAnalysis.model_evaluations import get_rmse, get_percent_error
from Models.model_knn import get_trained_model as create_KNN, input_columns as KNN_columns
from Models.model_random_forest import get_trained_model as create_forest, input_columns as forest_columns
from helpers import to_millions
from matplotlib_format import slides_format, integer_axis


# Split error and quantiled are basically duplicates
def split_error(model, inputs_test: pd.DataFrame, outputs_test: pd.DataFrame, levels=5, error_function=get_rmse):
    """
    Returns a salary level and the corresponding error
    """
    levels += 1
    
    combined = inputs_test.copy()
    combined["actual"] = outputs_test

    cutoffs = np.linspace(0, max(combined["actual"]), levels)

    errors = []

    prev_quantile = cutoffs[0]
    for quantile in cutoffs[1:]:
        selected = combined[(prev_quantile < combined["actual"]) & (combined["actual"] <= quantile)]
        print(prev_quantile, quantile)

        if len(selected) == 0:
            errors.append(0)
            print("Decrease the number of bins so that everyone has one, that way we can show error over the full range.")
            
            continue

        errors.append(error_function(model, selected.drop("actual", axis=1), selected["actual"]))

        prev_quantile = quantile

    return cutoffs, errors

def quantiled_error(model, inputs_test: pd.DataFrame, outputs_test: pd.DataFrame, levels=5, error_function=get_rmse):
    """
    Returns a salary level and the corresponding error
    """
    levels += 1
    
    combined = inputs_test.copy()
    combined["actual"] = outputs_test

    percentages = np.linspace(0, 1, levels)
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

        errors.append(error_function(model, selected.drop("actual", axis=1), selected["actual"]))

    return quantiles, errors


# TODO: split into different functions for labeling and unit conversions purposes
def plot_error_over_salary(model1, model2, error_function, levels=5, quantiled=False):
    """Compares the error of two different model across different brackets of salaries."""

    if quantiled:
        quantiles, errors_KNN = quantiled_error(model1, get_inputs_test(model1.inputs), get_outputs_test(), error_function=error_function, levels=levels)
        
        # We already assigned quantiles
        _, errors_forest = quantiled_error(model2, get_inputs_test(model2.inputs), get_outputs_test(), error_function=error_function, levels=levels)

        labels = list(map(lambda num : f"{num:.1f}", to_millions(quantiles)))[:-1]
    else:
        # Do it linearly.
        cutoffs, errors_KNN = split_error(model1, get_inputs_test(model1.inputs), get_outputs_test(), error_function=error_function, levels=levels)
        
        # We already assigned cutoffs
        _, errors_forest = split_error(model2, get_inputs_test(model2.inputs), get_outputs_test(), error_function=error_function, levels=levels)

        labels = list(map(lambda num : f"{num:.1f}", to_millions(cutoffs)))[:-1]

    
    # Plot the bar charts side by side.
    x_axis = np.arange(len(errors_KNN))
    fig, ax = plt.subplots()

    errors_forest = to_millions(errors_forest)
    errors_KNN = to_millions(errors_KNN)

    width = 0.4
    for i, x in enumerate(x_axis):
        ax.bar(x - width/2, errors_KNN[i], width=width, color="tab:orange")
        ax.bar(x + width/2, errors_forest[i], width=width, color="tab:blue")

    plt.xticks(x_axis, labels=labels)

    plt.title("Model Performance vs Salary")
    plt.ylabel("Percent Error (%)")
    plt.xlabel("Salary Range (millions)")

    integer_axis(plt.gca())
    plt.legend(["KNN", "Random forest"])
    plt.ylabel("RMSE (millions)")
    plt.title("RMSE Linear")

def compare_error_across_salaries():
    plot_error_over_salary(create_KNN(), create_forest(), get_percent_error, quantiled=True)
    plt.legend(["KNN", "Random forest"])
    plt.title("Percent Error Quantiled")
    plt.show()

    plot_error_over_salary(create_KNN(), create_forest(), get_percent_error, levels=7, quantiled=False)
    plt.legend(["KNN", "Random forest"])
    plt.title("Percent Error Linear")
    plt.show()

    plot_error_over_salary(create_KNN(), create_forest(), get_rmse, quantiled=True)
    plt.legend(["KNN", "Random forest"])
    plt.ylabel("RMSE (millions)")
    plt.title("RMSE Quantiled")
    plt.show()

    plot_error_over_salary(create_KNN(), create_forest(), get_rmse, levels=7, quantiled=False)
    plt.legend(["KNN", "Random forest"])
    plt.ylabel("RMSE (millions)")
    plt.title("RMSE Linear")
    plt.show()

def top_error(model, n=10):
    players = get_player_data().copy()
    players["prediction"] = model.predict(players[model.input_columns])
    players["error"] = np.abs(players["prediction"]  - get_outputs_test())

    players = players.sort_values("error", ascending=False)

    players['adjusted_career_revenue'] = to_millions(players['adjusted_career_revenue'])
    players['error'] = to_millions(players['error'])
    players['prediction'] = to_millions(players['prediction'])

    for index, player in players.head(n).iterrows():
        if player["prediction"] > player["adjusted_career_revenue"]:
            print(f"{player['name']:20s}: Overpredicted  {player['adjusted_career_revenue']:5.0f} by {player['error']:5.0f}")
        else:
            print(f"{player['name']:20s}: Underpredicted {player['adjusted_career_revenue']:5.0f} by {player['error']:5.0f}")

def get_residuals(model, in_millions: bool = False):
    players = get_player_data().copy()
    players["prediction"] = model.predict(players[model.input_columns])
    players["error"] = get_outputs_test() - players["prediction"]

    if in_millions:
        players["error"] = to_millions(players["error"])

    return players["error"]

def plot_residuals(model, bins=40):
    residuals = get_residuals(model, in_millions=True)

    plt.hist(residuals, bins=bins)
    plt.title("Residuals Trend")
    plt.xlabel("Residual")
    plt.ylabel("Frequency")

def print_residual_distribution(model):
    residuals = get_residuals(model, in_millions=True)
    print(f"Mean: {np.mean(residuals)}")
    print(f"STD: {np.std(residuals)}")

def compare_position(model, position):
    players = get_player_data().copy()
    players["error"] = get_residuals(model, in_millions=True)

    players_grouped = players.groupby(position)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_dpi(200)

    ax1.bar(["Not", position], players_grouped["error"].mean())
    ax1.set_title(f"Bias of {position}")
    ax1.set_ylabel("Average Residual (millions)")

    relevant_players = players[players[position] == True]
    in_position = get_rmse(model, relevant_players[model.input_columns], relevant_players["adjusted_career_revenue"])
    relevant_players = players[players[position] == False]
    not_position = get_rmse(model, relevant_players[model.input_columns], relevant_players["adjusted_career_revenue"])

    rmse_values = to_millions([not_position, in_position])

    ax2.bar(["Not", position], rmse_values)
    ax2.set_title(f"Error of {position}")
    ax2.set_ylabel("RMSE (millions)")


if __name__ == "__main__":
    slides_format()
    # compare_error_across_salaries()

    # plot_error_over_salary(create_KNN(), create_forest(), get_rmse, levels=25, quantiled=False)
    # top_error(create_forest())
    # top_error(create_KNN())

    # plot_residuals(create_forest())
    # print_residual_distribution(create_forest())

    # plot_residuals(create_KNN())
    # print_residual_distribution(create_KNN())
    
    for position in positions:
        compare_position(create_forest(), position)
        plt.show()

    # compare_position(create_KNN(), "Small Forward")
    # plt.show()

    pass