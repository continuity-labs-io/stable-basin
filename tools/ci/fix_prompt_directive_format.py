import os
import re

directory = "docs/ai/audited"
pattern = re.compile(r'^([A-Z0-9\s_-]+):\s*(.*)$')

for filename in os.listdir(directory):
    if not filename.endswith(".md"):
        continue
    filepath = os.path.join(directory, filename)
    with open(filepath, "r") as f:
        lines = f.readlines()
        
    new_lines = []
    modified = False
    for line in lines:
        match = pattern.match(line)
        if match:
            heading_text = match.group(1).strip()
            rest = match.group(2).strip()
            
            # Must contain at least one uppercase letter and not just be a short word like 'A'
            if re.search(r'[A-Z]', heading_text) and len(heading_text) > 2:
                modified = True
                new_lines.append(f"## {heading_text}\n")
                if rest:
                    new_lines.append(f"{rest}\n")
                continue
                
        new_lines.append(line)
        
    if modified:
        with open(filepath, "w") as f:
            f.writelines(new_lines)
        print(f"Modified {filename}")
