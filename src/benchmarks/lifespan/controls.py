import numpy as np

POSITIVE_MIN_C = 0.65
NEGATIVE_BAND = (0.40, 0.60)
LEAK_MIN_C = 0.75

def check_controls(results: dict) -> list[str]:
    """
    results should contain the C-indices of the controls.
    e.g. {'positive_F2': c_mean, 'negative_F1': c_mean, 'negative_F2': c_mean, 'negative_F1+F2': c_mean, ...}
    """
    failures = []
    
    if 'positive_F2' in results:
        val = results['positive_F2']
        if val < POSITIVE_MIN_C:
            failures.append(f"Positive F2 C = {val:.3f} < {POSITIVE_MIN_C}")
            
    for k in ['negative_F1', 'negative_F2', 'negative_F1+F2']:
        if k in results:
            val = results[k]
            if not (NEGATIVE_BAND[0] <= val <= NEGATIVE_BAND[1]):
                failures.append(f"Negative {k.replace('negative_', '')} C = {val:.3f} outside {NEGATIVE_BAND}")
                
    if 'leak_F1+F2' in results:
        val = results['leak_F1+F2']
        if val < LEAK_MIN_C:
            failures.append(f"Leak F1+F2 C = {val:.3f} < {LEAK_MIN_C}")
            
    return failures
