import pandas as pd
from path_config import *

if __name__ == "__main__":
    df = pd.read_csv(INFLATION_PATH)

    df["year"] = 0
    for index, row in df.iterrows():        
        year = int(df.loc[index, "DATE"].split("-")[0])

        df.loc[index, "year"] = year
    
    # 1984 is when the first salary is from
    index = df[df["year"] == 1984].first_valid_index()
    multiplier = 100 / df.iloc[index]["GDPDEF"]

    df["inflation"] = df["GDPDEF"] * multiplier

    df.to_csv(CLEANED_INFLATION_PATH)