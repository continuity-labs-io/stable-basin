import time
import math
import sys
import os

# ANSI Color Codes
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_slow(text, delay=0.03):
    """Prints text slowly for dramatic terminal effect."""
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()

def get_patient_journal(ksm):
    if ksm > 0.85:
        return Colors.GREEN + "Patient Journal: 'I feel incredible! The recovery was a breeze. I'm going to live forever.'" + Colors.ENDC
    elif ksm > 0.70:
        return Colors.CYAN + "Patient Journal: 'Still seeing great results, but my joints ached for a few weeks this time.'" + Colors.ENDC
    elif ksm > 0.50:
        return Colors.YELLOW + "Patient Journal: 'The slog is real. It took months to get out of bed. Is it still working?'" + Colors.ENDC
    elif ksm > 0.30:
        return Colors.RED + "Patient Journal: 'I'm trapped. The therapy nearly killed me. I look in the mirror and nothing changed.'" + Colors.ENDC
    else:
        return Colors.RED + Colors.BOLD + "Patient Journal: '... [Patient unresponsive due to extreme biological shock]'" + Colors.ENDC

def get_color_for_ksm(ksm):
    if ksm > 0.8:
        return Colors.GREEN
    elif ksm > 0.5:
        return Colors.YELLOW
    return Colors.RED

def run_simulation():
    # Make output dir in case we write files later
    os.makedirs("output/demo", exist_ok=True)
    
    print(Colors.HEADER + Colors.BOLD + "="*60)
    print("      THE RATCHET SIMULATOR: BIOLOGICAL PLATEAU DEMO")
    print("="*60 + Colors.ENDC)
    print("Simulating 20 consecutive years of Level 3 Rejuvenation Therapy...")
    print("Patient Starting Age: 50 | Base KSM (Stability): 0.95\n")
    time.sleep(1)

    # Initial State
    bio_age = 50.0
    ksm = 0.95
    
    # Simulation Parameters
    max_rejuvenation_power = 5.0  # Max years knocked off when KSM is 1.0
    ksm_degradation_per_shock = 0.035  # The irreversible thermodynamic damage per therapy
    
    for year in range(1, 21):
        # 1. Natural aging (time passes before therapy)
        bio_age += 1.0
        
        # 2. The Therapy Application
        # How much age do we knock off? It's gated by the intrinsic plasticity (KSM)
        age_reduction = max_rejuvenation_power * (ksm ** 1.5)
        bio_age -= age_reduction
        
        # 3. The Consequences (Thermodynamic Noise degrades the system)
        ksm = max(0.01, ksm - ksm_degradation_per_shock)
        
        # 4. Recovery Time (Explodes exponentially as KSM drops)
        # At KSM=0.95, recovery is ~11 days. At KSM=0.2, recovery is ~250 days.
        recovery_days = int(10.0 / (ksm ** 2))
        if recovery_days > 365:
            recovery_days = 365
            
        # UI Output
        color = get_color_for_ksm(ksm)
        journal = get_patient_journal(ksm)
        
        # Format the numbers nicely
        age_str = f"{bio_age:.1f}"
        ksm_str = f"{ksm:.2f}"
        reduction_str = f"{-age_reduction:.1f} yrs"
        
        print_slow(f"{Colors.BOLD}Year {year:02d}{Colors.ENDC} | Therapy Administered")
        print(f"  └─ KSM (Stability): {color}{ksm_str}{Colors.ENDC}")
        print(f"  └─ Age Reduction:   {color}{reduction_str}{Colors.ENDC}")
        print(f"  └─ Recovery Time:   {color}{recovery_days} days{Colors.ENDC}")
        print(f"  └─ Biological Age:  {Colors.BOLD}{age_str}{Colors.ENDC}")
        print(f"  └─ {journal}\n")
        
        # Dramatic pause
        time.sleep(0.4)

    print(Colors.HEADER + Colors.BOLD + "="*60)
    print("                     SIMULATION COMPLETE")
    print("="*60 + Colors.ENDC)
    print_slow("CONCLUSION: The 'Ratchet Effect' demonstrates that without addressing", delay=0.02)
    print_slow("the underlying loss of Koopman Stability Margin (KSM), repeated", delay=0.02)
    print_slow("rejuvenation therapies experience diminishing returns, eventually", delay=0.02)
    print_slow("resulting in an inescapable asymptotic biological plateau.", delay=0.02)

if __name__ == "__main__":
    run_simulation()
