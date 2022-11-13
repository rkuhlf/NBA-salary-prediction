import pandas as pd

from helpers import na_count


df = pd.read_csv("salaries_1985to2018.csv")

print(na_count(df))

df.dropna(subset=["team"], inplace=True)

print(na_count(df))


df_inflation = pd.read_csv("cleaned_inflation.csv")
adjusted_salaries = []
for index, row in df.iterrows():
    year = row["season_start"]
    index = df_inflation[df_inflation["year"] == year].first_valid_index()
    inflation = df_inflation.iloc[index]["inflation"] / 100

    adjusted_salaries.append(df.loc[index, "salary"] / inflation)

df["adjusted_salary"] = adjusted_salaries

df.to_csv("cleaned_salaries.csv", index=False)