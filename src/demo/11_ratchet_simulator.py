import time
import math
import sys
import os
import plotext as plt

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

def get_patient_journal(bio_age):
    if bio_age > 70:
        return Colors.RED + "Patient Journal: 'The procedure nearly killed me. I was bedridden for months. I feel so frail.'" + Colors.ENDC
    elif bio_age > 50:
        return Colors.YELLOW + "Patient Journal: 'It was a slog, but I'm recovering faster than before. I can feel my joints loosening.'" + Colors.ENDC
    elif bio_age > 35:
        return Colors.CYAN + "Patient Journal: 'I bounced back in just a few weeks! I'm running again. This is incredible.'" + Colors.ENDC
    else:
        return Colors.GREEN + "Patient Journal: 'I feel invincible! The recovery was barely a weekend. I am getting my life back!'" + Colors.ENDC

def get_color_for_age(bio_age):
    if bio_age > 70:
        return Colors.RED
    elif bio_age > 50:
        return Colors.YELLOW
    elif bio_age > 35:
        return Colors.CYAN
    return Colors.GREEN

def run_interactive_simulation():
    os.makedirs("output/demo", exist_ok=True)
    
    while True:
        print(Colors.HEADER + Colors.BOLD + "="*60)
        print("      THE RATCHET SIMULATOR: BENJAMIN BUTTON DEMO")
        print("="*60 + Colors.ENDC)
        
        # 1. Intake Prompt
        try:
            start_age = float(input(Colors.BOLD + "Enter the patient's starting age: " + Colors.ENDC))
        except ValueError:
            print(Colors.RED + "Invalid input. Exiting." + Colors.ENDC)
            return
            
        print()
        time.sleep(0.5)

        # 2. Cryopreservation Threshold
        if start_age >= 90:
            print(Colors.RED + Colors.BOLD + "WARNING: EXTREME FRAILTY INDEX DETECTED." + Colors.ENDC)
            print_slow("Patient's biological structures cannot survive the physical shock of Level 3 Rejuvenation.")
            print_slow("Recommendation: Immediate Cryopreservation until Level 4 (in-situ cellular reprogramming) is available.")
            print(Colors.BOLD + "Simulation Aborted." + Colors.ENDC)
        else:
            print_slow("Patient viable. Initiating Level 3 Rejuvenation Therapy protocol...")
            print_slow("Goal: Reach a Biological Age of 20.\n")
            
            chrono_age = start_age
            bio_age = start_age
            
            cycle = 1
            years_between_treatments = 2.0
            total_recovery_days = 0
            
            history_years = [0.0]
            history_bio = [bio_age]
            history_chrono = [chrono_age]
            current_year = 0.0
            
            while bio_age > 20:
                print(Colors.HEADER + f"\n--- Clinic Visit #{cycle} ---" + Colors.ENDC)
                
                # ASCII Plot
                plt.clear_figure()
                plt.plot(history_years, history_chrono, label="Chrono Age", color="blue", marker="dot")
                plt.plot(history_years, history_bio, label="Bio Age", color="green", marker="dot")
                plt.title(f"Age Trajectory (Year {current_year})")
                plt.xlabel("Years Since Start")
                plt.ylabel("Age")
                plt.plot_size(60, 15)
                plt.show()
                print()
                
                print(f"Current Chronological Age: {Colors.BOLD}{chrono_age:.1f}{Colors.ENDC}")
                print(f"Current Biological Age:    {get_color_for_age(bio_age)}{bio_age:.1f}{Colors.ENDC}\n")
                
                # Interactive prompt
                print(Colors.BOLD + "Select Treatment Plan:" + Colors.ENDC)
                print("  [1] Conservative (Low Impact, Fast Recovery)")
                print("  [2] Recommended  (Standard Impact, Normal Recovery)")
                print("  [3] Aggressive   (High Impact, Brutal Recovery)")
                print("  [0] Refuse Treatment")
                
                choice = input(Colors.BOLD + "Enter choice [0-3]: " + Colors.ENDC).strip()
                
                if choice == '0':
                    print("Patient refused treatment. They will continue to age naturally.")
                    break
                elif choice == '1':
                    therapy_power = 2.0
                    recovery_mult = 0.5
                elif choice == '3':
                    therapy_power = 12.0
                    recovery_mult = 2.5
                else:
                    # Default to recommended for '2' or any invalid input
                    therapy_power = 6.0
                    recovery_mult = 1.0
                    
                print("\nAdministering therapy...")
                time.sleep(0.5)
                
                # The Math: Recovery time is an exponential curve based on biological age
                base_recovery = 2.0 * math.exp(0.06 * bio_age)
                recovery_days = int(base_recovery * recovery_mult)
                
                if recovery_days > 365:
                    print(Colors.RED + Colors.BOLD + "\nTHERAPY REJECTED" + Colors.ENDC)
                    print_slow(f"Projected recovery time is {recovery_days} days.")
                    print_slow("Clinic policy prohibits treatments requiring >1 year of continuous bed rest.")
                    print_slow("The patient cannot tolerate the procedure and will age naturally.")
                    break
                    
                total_recovery_days += recovery_days
                
                # Therapy succeeds
                bio_age -= therapy_power
                if bio_age < 20:
                    bio_age = 20.0
                    
                color = get_color_for_age(bio_age)
                journal = get_patient_journal(bio_age)
                
                print_slow(f"  └─ Recovery Time:  {color}{recovery_days} days bedridden{Colors.ENDC}", delay=0.01)
                print_slow(f"  └─ Post-op BioAge: {color}{bio_age:.1f} years{Colors.ENDC}", delay=0.01)
                print_slow(f"  └─ {journal}\n", delay=0.01)
                
                if bio_age <= 20:
                    print(Colors.GREEN + Colors.BOLD + "="*60)
                    print("                     VICTORY ACHIEVED")
                    print("="*60 + Colors.ENDC)
                    print_slow("The patient has successfully achieved a biological age of 20.")
                    print_slow(f"Final Chronological Age: {chrono_age:.1f} years old.")
                    print_slow(f"Total Time Bedridden: {total_recovery_days} days ({total_recovery_days/365.25:.1f} years)")
                    print_slow("They have achieved functional immortality. Welcome to the future.")
                    break
                    
                print(Colors.CYAN + f"Doctor's orders: 'Come back in {int(years_between_treatments)} years for your next round.'" + Colors.ENDC)
                print("Time passes...\n")
                time.sleep(1)
                
                # Time Skip
                chrono_age += years_between_treatments
                bio_age += years_between_treatments
                current_year += years_between_treatments
                history_years.append(current_year)
                history_chrono.append(chrono_age)
                history_bio.append(bio_age)
                cycle += 1
                
        print("\n" + "="*60)
        play_again = input(Colors.BOLD + "Would you like to simulate another patient? [Y/n]: " + Colors.ENDC).strip().lower()
        if play_again == 'n':
            print("Exiting simulator. Have a long and healthy life!")
            break
        print("\n\n")

if __name__ == "__main__":
    try:
        run_interactive_simulation()
    except KeyboardInterrupt:
        print("\nSimulation aborted.")
