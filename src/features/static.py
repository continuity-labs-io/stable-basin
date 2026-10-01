import pandas as pd

def static(rows: pd.DataFrame, variables: list[str]) -> pd.Series:
    daily_means = rows.groupby('age_days')[variables].mean()
    f1_mean = daily_means.mean(axis=0)
    f1_sd = daily_means.std(axis=0)
    
    idx = [f"{v}_mean" for v in variables] + [f"{v}_sd" for v in variables]
    return pd.Series(pd.concat([f1_mean, f1_sd]).values, index=idx)
