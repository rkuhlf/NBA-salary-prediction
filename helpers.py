import numpy as np


def na_count(df):
    na_count = df.isna().sum().sum()

    return na_count

def to_millions(arr):
    return np.array(arr) / 1e6