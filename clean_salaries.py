import pandas as pd

from helpers import na_count


df = pd.read_csv("salaries_1985to2018.csv")

print(na_count(df))

df.dropna(subset=["team"], inplace=True)

print(na_count(df))


df_inflation = pd.read_csv("cleaned_inflation.csv")

df["adjusted_salary"] = 0
for index, row in df_inflation.iterrows():
    year = row["year"]
    index = df_inflation[df_inflation["year"] == year].first_valid_index()
    inflation = df_inflation.iloc[index]["inflation"] / 100

    df.loc[df["season_start"] == year, "adjusted_salary"] = df.loc[df["season_start"] == year, "salary"] / inflation


df.to_csv("cleaned_salaries.csv", index=False)