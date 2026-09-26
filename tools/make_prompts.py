import argparse
import os
import glob
import re

PREFIX_FILE = "docs/ai/.prompt_prefix"
AUDIT_DIR = "docs/ai/audit"

def get_last_prefix():
    if os.path.exists(PREFIX_FILE):
        with open(PREFIX_FILE, 'r') as f:
            content = f.read().strip()
            if content.isdigit():
                return int(content)
    
    # Fallback to discovering the max prefix in the directory
    max_prefix = 0
    for file_path in glob.glob("docs/ai/**/*.md", recursive=True):
        basename = os.path.basename(file_path)
        match = re.match(r'^(\d{3})_', basename)
        if match:
            prefix = int(match.group(1))
            if prefix > max_prefix:
                max_prefix = prefix
    
    return max_prefix

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--number", type=int, default=1, help="Number of prompts to generate")
    args = parser.parse_args()

    last_prefix = get_last_prefix()
    os.makedirs(AUDIT_DIR, exist_ok=True)
    
    new_last_prefix = last_prefix
    for i in range(1, args.number + 1):
        current_prefix = last_prefix + i
        new_last_prefix = current_prefix
        
        filename = f"{current_prefix:03d}_.md"
        filepath = os.path.join(AUDIT_DIR, filename)
        
        if not os.path.exists(filepath):
            with open(filepath, 'w') as f:
                f.write("\n")
            print(f"Created {filepath}")
        else:
            print(f"File {filepath} already exists. Skipping.")
    
    # Update prefix file
    with open(PREFIX_FILE, 'w') as f:
        f.write(str(new_last_prefix) + "\n")
    print(f"Updated {PREFIX_FILE} to {new_last_prefix}")

if __name__ == "__main__":
    main()
