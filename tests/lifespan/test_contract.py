import pandas as pd
import pytest
from src.benchmarks.lifespan.contract import validate

def test_contract_validates_ok():
    animals = pd.DataFrame({'animal_id': ['a1', 'a2'], 'lifespan_days': [10.0, 20.0]})
    series = pd.DataFrame({'animal_id': ['a1', 'a1', 'a2'], 'age_days': [1, 2, 1], 'v1': [1.0, 2.0, 3.0]})
    validate(animals, series, ['v1'], set())

def test_contract_duplicate_animal():
    animals = pd.DataFrame({'animal_id': ['a1', 'a1'], 'lifespan_days': [10.0, 20.0]})
    series = pd.DataFrame({'animal_id': ['a1'], 'age_days': [1], 'v1': [1.0]})
    with pytest.raises(ValueError, match="animal_id is not unique in animals."):
        validate(animals, series, ['v1'], set())

def test_contract_missing_animal():
    animals = pd.DataFrame({'animal_id': ['a1'], 'lifespan_days': [10.0]})
    series = pd.DataFrame({'animal_id': ['a1', 'a2'], 'age_days': [1, 1], 'v1': [1.0, 2.0]})
    with pytest.raises(ValueError, match="series animal_id is missing from animals."):
        validate(animals, series, ['v1'], set())

def test_contract_missing_variable():
    animals = pd.DataFrame({'animal_id': ['a1'], 'lifespan_days': [10.0]})
    series = pd.DataFrame({'animal_id': ['a1'], 'age_days': [1], 'v1': [1.0]})
    with pytest.raises(ValueError, match="Variable v2 missing from series."):
        validate(animals, series, ['v1', 'v2'], set())

def test_contract_non_numeric():
    animals = pd.DataFrame({'animal_id': ['a1'], 'lifespan_days': [10.0]})
    series = pd.DataFrame({'animal_id': ['a1'], 'age_days': [1], 'v1': ['foo']})
    with pytest.raises(ValueError, match="Variable v1 is not numeric in series."):
        validate(animals, series, ['v1'], set())

def test_contract_forbidden_variable():
    animals = pd.DataFrame({'animal_id': ['a1'], 'lifespan_days': [10.0]})
    series = pd.DataFrame({'animal_id': ['a1'], 'age_days': [1], 'died': [1.0]})
    with pytest.raises(ValueError, match="Forbidden variable listed: died"):
        validate(animals, series, ['died'], {'died'})

def test_contract_age_exceeds_lifespan():
    animals = pd.DataFrame({'animal_id': ['a1'], 'lifespan_days': [10.0]})
    series = pd.DataFrame({'animal_id': ['a1'], 'age_days': [12], 'v1': [1.0]})
    with pytest.raises(ValueError, match="age_days exceeds lifespan_days \\+ 1."):
        validate(animals, series, ['v1'], set())
