import pandas as pd

def cut_at_landmark(animals: pd.DataFrame, series: pd.DataFrame, variables: list[str], L: int, min_valid_days: int, min_valid_bins: int = 120):
    """
    Censors the dataset at a specific landmark age and filters animals based on valid days.

    Args:
        animals (pd.DataFrame): DataFrame containing one row per animal with `animal_id` and `lifespan_days`.
        series (pd.DataFrame): DataFrame containing time-series measurements per animal and time bin.
        variables (list[str]): A list of column names in `series` representing the measured variables.
        L (int): The landmark age (in days) to censor at.
        min_valid_days (int): Minimum number of valid days required before the landmark age for an animal to be included.
        min_valid_bins (int, optional): Minimum number of valid bins (all variables present) required for a day to be considered valid. Defaults to 120.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: A tuple containing:
            - pre_series (pd.DataFrame): The filtered time-series data containing only valid days before the landmark age.
            - outcomes (pd.DataFrame): A DataFrame of survival outcomes with columns `animal_id`, `T` (remaining lifespan), `E` (event occurred), and `group`.
    """
    series_valid = series.dropna(subset=variables)
    
    pre_L_valid = series_valid[series_valid['age_days'] < L]
    
    daily_bins = pre_L_valid.groupby(['animal_id', 'age_days']).size()
    valid_days = daily_bins[daily_bins >= min_valid_bins].reset_index()
    
    days_per_animal = valid_days.groupby('animal_id').size()
    
    valid_animal_ids = set(days_per_animal[days_per_animal >= min_valid_days].index)
    
    animals_filtered = animals[animals['lifespan_days'] >= L].copy()
    animals_filtered = animals_filtered[animals_filtered['animal_id'].isin(valid_animal_ids)]
    
    valid_animal_ids_final = set(animals_filtered['animal_id'])
    
    valid_day_keys = set(zip(valid_days['animal_id'], valid_days['age_days']))
    
    # According to prompt: "pre_series = valid days with age_days < L only"
    # Keep only rows where (animal_id, age_days) is in valid_day_keys and animal is valid
    pre_series = series[series.set_index(['animal_id', 'age_days']).index.isin(valid_day_keys)]
    pre_series = pre_series[pre_series['animal_id'].isin(valid_animal_ids_final)].copy()
    
    if len(pre_series) > 0:
        assert pre_series['age_days'].max() < L
        
    outcomes = pd.DataFrame({
        'animal_id': animals_filtered['animal_id'],
        'T': animals_filtered['lifespan_days'] - L,
        'E': animals_filtered['died'],
        'group': animals_filtered['group'] if 'group' in animals_filtered.columns else 'unknown'
    })
    
    return pre_series, outcomes
