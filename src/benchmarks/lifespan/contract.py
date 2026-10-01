import pandas as pd

def validate(animals: pd.DataFrame, series: pd.DataFrame, variables: list[str], forbidden_variables: set[str]):
    """
    Validates the data contract for the lifespan benchmark inputs.

    Args:
        animals (pd.DataFrame): DataFrame containing one row per animal. Must include 
            `animal_id` (unique) and `lifespan_days`.
        series (pd.DataFrame): DataFrame containing time-series measurements per animal and time bin. 
            Must include `animal_id`, `age_days`, and the variables specified in `variables`.
        variables (list[str]): A list of column names in `series` representing the measured variables.
        forbidden_variables (set[str]): A set of variable names that are forbidden to be used as features.

    Raises:
        ValueError: If any contract constraint is violated (e.g., duplicate animals, missing 
            animals in series, missing/non-numeric variables, forbidden variables, or invalid ages).

    Returns:
        None
    """
    if not animals['animal_id'].is_unique:
        raise ValueError("animal_id is not unique in animals.")
    
    missing_animals = set(series['animal_id']) - set(animals['animal_id'])
    if missing_animals:
        raise ValueError("series animal_id is missing from animals.")
        
    for var in variables:
        if var not in series.columns:
            raise ValueError(f"Variable {var} missing from series.")
        if not pd.api.types.is_numeric_dtype(series[var]):
            raise ValueError(f"Variable {var} is not numeric in series.")
            
    for var in variables:
        if var in forbidden_variables:
            raise ValueError(f"Forbidden variable listed: {var}")
            
    merged = pd.merge(series, animals[['animal_id', 'lifespan_days']], on='animal_id')
    if (merged['age_days'] > merged['lifespan_days'] + 1).any():
        raise ValueError("age_days exceeds lifespan_days + 1.")
