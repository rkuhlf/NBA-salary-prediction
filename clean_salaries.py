import pandas as pd

from helpers import na_count


df = pd.read_csv("salaries_1985to2018.csv")

print(na_count(df))

df.dropna(subset=["team"], inplace=True)

print(na_count(df))

df.to_csv("cleaned_salaries.csv", index=False)