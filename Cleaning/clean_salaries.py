import pandas as pd

from config import *
from helpers import na_count


df = pd.read_csv(SALARIES_PATH)

print(na_count(df))
df.dropna(subset=["team"], inplace=True)
print(na_count(df))


df_inflation = pd.read_csv(CLEANED_INFLATION_PATH)

# Edit salary dataframe to have an additional column that adjusts for inflation based on the GDP deflator.
df["adjusted_salary"] = 0
for index, row in df_inflation.iterrows():
    year = row["year"]
    index = df_inflation[df_inflation["year"] == year].first_valid_index()
    inflation = df_inflation.iloc[index]["inflation"] / 100

    df.loc[df["season_start"] == year, "adjusted_salary"] = df.loc[df["season_start"] == year, "salary"] / inflation


df.to_csv(CLEANED_SALARIES_PATH, index=False)