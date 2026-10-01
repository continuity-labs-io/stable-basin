import pandas as pd
import numpy as np
from pathlib import Path
from src.data.behavior.synthetic_aging import ou_process

def killifish_bedbrook(cfg):
    path = Path(cfg.get('path'))
    cache_file = Path('data/cache/killifish_bedbrook.parquet')
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    
    if not cache_file.exists():
        # First read a few rows to get columns
        df_head = pd.read_csv(path, nrows=3)
        if 'snout_velocity' in df_head.columns:
            feature_cols = ['snout_velocity', 'snout_acceleration', 'disp', 'body_length', 'count_snout', 'active', 'inactive', 'sleep']
        else:
            feature_cols = [c for c in df_head.columns if c.startswith('state_')]
            
        meta_cols = ['datetime', 'full_fish_name', 'fish_number', 'age_days', 'lifespan', 'status', 'genotype', 'sex', 'feeding', 'cohort']
        needed_cols = [c for c in feature_cols + meta_cols if c in df_head.columns]
        
        chunks = []
        for chunk in pd.read_csv(path, usecols=needed_cols, chunksize=100000):
            if 'feature' in chunk.columns:
                idx_cols = [c for c in chunk.columns if c not in ['feature', 'value']]
                chunk = chunk.pivot(index=idx_cols, columns='feature', values='value').reset_index()
            chunks.append(chunk)
        df = pd.concat(chunks, ignore_index=True)
        df.to_parquet(cache_file)
    else:
        df = pd.read_parquet(cache_file)
        if 'snout_velocity' in df.columns:
            feature_cols = ['snout_velocity', 'snout_acceleration', 'disp', 'body_length', 'count_snout', 'active', 'inactive', 'sleep']
        else:
            feature_cols = [c for c in df.columns if c.startswith('state_')]
            
    filters = cfg.get('filters', {})
    for k, v in filters.items():
        if k in df.columns:
            df = df[df[k] == v]
            
    df['animal_id'] = df['full_fish_name']
    df['group'] = df['cohort'] if 'cohort' in df.columns else 'unknown'
    df['died'] = (df['status'] == 'd').astype(int)
    
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['bin'] = (df['datetime'].dt.hour * 60 + df['datetime'].dt.minute) // 10
    
    if cfg.get('negative_is_missing', True):
        for c in feature_cols:
            df.loc[df[c] < 0, c] = np.nan
            
    fish_order = df['animal_id'].drop_duplicates().values
    
    # A bin is valid if all variables are present. (Not filtering out rows, but variables list is feature_cols).
    
    animals = df[['animal_id', 'lifespan', 'died', 'group']].drop_duplicates('animal_id')
    animals = animals.rename(columns={'lifespan': 'lifespan_days'})
    
    # Reorder animals to match first appearance
    animals = animals.set_index('animal_id').loc[fish_order].reset_index()
    
    series_cols = ['animal_id', 'age_days', 'bin'] + feature_cols
    series = df[series_cols]
    
    variables = feature_cols
    return animals, series, variables

def _generate_synthetic(cfg, mode="positive"):
    rng = np.random.default_rng(cfg.get('seed', 42))
    n_animals = 80
    bins_per_day = 144
    n_vars = 4
    
    animals_list = []
    series_list = []
    
    for i in range(n_animals):
        animal_id = f"anim_{i}"
        
        if mode == "positive":
            tau_i = rng.uniform(1, 10)
            T = 60 - 5 * tau_i + rng.normal(0, 5)
            T = max(5, T)
            lifespan_days = int(60 + T)
            tau_array = np.full(lifespan_days * bins_per_day, tau_i * bins_per_day)
            
        died = 1 if rng.random() > 0.3 else 0
        group = "synth"
        
        animals_list.append({
            'animal_id': animal_id,
            'lifespan_days': float(lifespan_days),
            'died': died,
            'group': group
        })
        
        time_of_day = np.arange(lifespan_days * bins_per_day) % bins_per_day
        profile = np.sin(2 * np.pi * time_of_day / bins_per_day)
        
        df_series = pd.DataFrame({
            'animal_id': animal_id,
            'age_days': np.arange(lifespan_days * bins_per_day) // bins_per_day,
            'bin': time_of_day
        })
        
        for v in range(n_vars):
            x = ou_process(len(tau_array), 1.0, tau_array, sigma=1.0, rng=rng)
            # Add shared 24h profile
            x += profile + v * 10
            df_series[f'v{v}'] = x
            
        series_list.append(df_series)
        
    animals = pd.DataFrame(animals_list)
    series = pd.concat(series_list, ignore_index=True)
    variables = [f'v{v}' for v in range(n_vars)]
    
    return animals, series, variables

def synthetic_positive(cfg):
    return _generate_synthetic(cfg, mode="positive")

def synthetic_negative_control(cfg):
    """Data before the landmark carry no information about lifespan. A score above chance
    means post-landmark data reached the features."""
    rng = np.random.default_rng(cfg.get('seed', 42))
    L = cfg.get('landmark', 60)
    n = cfg.get('n_animals', 100)
    lifespan = rng.uniform(L + 20, L + 80, n).round()
    died = rng.random(n) < 0.7
    group = 'synth'

    animals_list = []
    series_list = []
    
    lifespan_z = (lifespan - lifespan.mean()) / lifespan.std()
    
    for i in range(n):
        animal_id = f"anim_{i}"
        animals_list.append({
            'animal_id': animal_id,
            'lifespan_days': float(lifespan[i]),
            'died': int(died[i]),
            'group': group
        })
        
        T_days = int(lifespan[i])
        n_bins = T_days * 144
        
        time_of_day = np.arange(n_bins) % 144
        profile = np.sin(2 * np.pi * time_of_day / 144)
        
        df_series = pd.DataFrame({
            'animal_id': animal_id,
            'age_days': np.arange(n_bins) // 144,
            'bin': time_of_day
        })
        
        z_i = lifespan_z[i]
        
        for v in range(4):
            x = ou_process(n_bins, 1.0, 2 * 144, 1.0, rng=rng)
            x += profile + v * 10
            
            mask = df_series['age_days'] >= L
            x[mask] += 3.0 * np.sqrt(0.5 * 2 * 144) * z_i
            
            df_series[f'v{v}'] = x
            
        series_list.append(df_series)
        
    animals = pd.DataFrame(animals_list)
    series = pd.concat(series_list, ignore_index=True)
    variables = [f'v{v}' for v in range(4)]
    
    return animals, series, variables

ADAPTERS = {
    'killifish_bedbrook': killifish_bedbrook,
    'synthetic_positive': synthetic_positive,
    'synthetic_negative_control': synthetic_negative_control
}
