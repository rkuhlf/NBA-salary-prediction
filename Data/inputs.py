"""
Helper code to ensure that all models use the same inputs and outputs, retrieving from the same file path and splitting by the same percentages.
"""


import pandas as pd
from sklearn.model_selection import train_test_split

from path_config import *



inflation_df = None
def get_inflation_data():
    global inflation_df

    if inflation_df is None:
        inflation_df = pd.read_csv(CLEANED_INFLATION_PATH)
    
    return inflation_df

salaries_df = None
def get_salary_data():
    global salaries_df 
    if salaries_df is None:
        salaries_df = pd.read_csv(CLEANED_SALARIES_PATH)
    
    return salaries_df

players_df = None
def get_player_data():
    global players_df 
    if players_df is None:
        players_df = pd.read_csv(CLEANED_PLAYERS_PATH)
    
    return players_df

def get_player_data_for_modeling(input_columns):
    players_df = get_player_data()
    
    inputs = players_df[input_columns]
    outputs = players_df["adjusted_career_revenue"]

    return inputs, outputs


FRAC_TEST = 0.3

def get_split_data(input_columns, random_state=101):
    inputs, outputs = get_player_data_for_modeling(input_columns)

    return train_test_split(inputs, outputs, test_size=FRAC_TEST, random_state=random_state)

def get_inputs_train(input_columns, random_state=101):
    inputs_train, inputs_test, outputs_train, outputs_test = get_split_data(input_columns, random_state=random_state)
    return inputs_train

def get_outputs_train(random_state=101):
    inputs_train, inputs_test, outputs_train, outputs_test = get_split_data([], random_state=random_state)
    return outputs_train

def get_inputs_test(input_columns, random_state=101):
    inputs_train, inputs_test, outputs_train, outputs_test = get_split_data(input_columns, random_state=random_state)
    return inputs_test

def get_outputs_test(random_state=101):
    inputs_train, inputs_test, outputs_train, outputs_test = get_split_data([], random_state=random_state)
    return outputs_test
