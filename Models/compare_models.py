
from matplotlib import pyplot as plt
import numpy as np
from Models.inputs import get_data

from Models.model_evaluations import compare_across_salaries
from Models.model_knn import get_trained_model as create_KNN, input_columns as KNN_columns
from Models.model_random_forest import get_trained_model as create_forest, input_columns as forest_columns

def to_millions(arr):
    return np.array(arr) / 1e6

def compare_error_over_salary():
    KNN_model, KNN_inputs, KNN_outputs = create_KNN()
    quantiles, errors_KNN = compare_across_salaries(KNN_model, KNN_columns, KNN_inputs, KNN_outputs)
    quantiles_labels = list(map(lambda num : f"{num:.1f}", to_millions(quantiles)))

    errors_KNN = to_millions(errors_KNN)
    x_axis = np.arange(len(errors_KNN))
    width = 0.4

    

    forest_model, forest_inputs, forest_outputs = create_forest()
    _, errors_forest = compare_across_salaries(forest_model, forest_columns, forest_inputs, forest_outputs)
    errors_forest = to_millions(errors_forest)

    fig, ax = plt.subplots()

    fig.set_dpi(200)

    for i, x in enumerate(x_axis):
        ax.bar(x - width/2, errors_KNN[i], width=width, color="tab:orange")
        ax.bar(x + width/2, errors_forest[i], width=width, color="tab:blue")

    ax.set_xticklabels(quantiles_labels)

    plt.title("Performance vs Salary")
    plt.ylabel("RMSE (millions)")
    plt.xlabel("Salary Range (millions)")
    plt.legend(["KNN", "Random forest"])
    
    plt.show()

if __name__ == "__main__":
    plt.rc('axes', titlesize=20)
    plt.rc('axes', labelsize=14)
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)
    # plt.rc('ylabel', labelsize=14)
    

    compare_error_over_salary()

    pass