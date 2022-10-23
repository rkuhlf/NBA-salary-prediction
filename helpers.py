def na_count(df):
    na_count = df.isna().sum().sum()

    return na_count