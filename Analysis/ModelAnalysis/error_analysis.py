"""Functions visualizing which parts of the data set have the most error and comparing the errors across different models."""

import math
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics

from Data.inputs import get_inputs_test, get_outputs_test
from Analysis.ModelAnalysis.model_evaluations import get_rmse, get_percent_error
from Models.model_knn import get_trained_model as create_KNN, input_columns as KNN_columns
from Models.model_random_forest import get_trained_model as create_forest, input_columns as forest_columns
from helpers import to_millions


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


def plot_error_over_salary(model1, model2, error_function, levels=5, quantiled=False):
    """Compares the error of two different model across different brackets of salaries."""

    if quantiled:
        quantiles, errors_KNN = quantiled_error(model1, get_inputs_test(model1.inputs), get_outputs_test(), error_function=error_function, levels=levels)
        
        # We already assigned quantiles
        _, errors_forest = quantiled_error(model2, get_inputs_test(model2.inputs), get_outputs_test(), error_function=error_function, levels=levels)

        labels = list(map(lambda num : f"{num:.1f}", to_millions(quantiles)))[:-1]
    else:
        # Do it linearly.
        cutoffs, errors_KNN = split_error(model1, get_inputs_test(model1.inputs), get_outputs_test(), error_function=error_function)
        
        # We already assigned cutoffs
        _, errors_forest = split_error(model2, get_inputs_test(model2.inputs), get_outputs_test(), error_function=error_function)

        labels = list(map(lambda num : f"{num:.1f}", to_millions(cutoffs)))[:-1]

    
    # Plot the bar charts side by side.
    x_axis = np.arange(len(errors_KNN))
    fig, ax = plt.subplots()

    width = 0.4
    for i, x in enumerate(x_axis):
        ax.bar(x - width/2, errors_KNN[i], width=width, color="tab:orange")
        ax.bar(x + width/2, errors_forest[i], width=width, color="tab:blue")

    plt.xticks(x_axis, labels=labels)

    plt.title("Model Performance vs Salary")
    plt.ylabel("Percent Error (%)")
    plt.xlabel("Salary Range (millions)")
    

if __name__ == "__main__":
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