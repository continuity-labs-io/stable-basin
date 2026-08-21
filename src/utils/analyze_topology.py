import subprocess
import os
import sys
import json
from collections import defaultdict

def main():
    target_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_md = os.path.join(script_dir, "topology_analysis.md")
    
    # Try to determine the package name (e.g. 'src')
    base_pkg = os.path.basename(os.path.abspath(target_dir))
    
    print(f"[*] Running pydeps dependency analysis on {target_dir} (Package: {base_pkg})...")
    try:
        # Run pydeps and capture the JSON dependency output
        result = subprocess.run(
            ["pydeps", target_dir, "--show-deps", "--no-output"], 
            capture_output=True, text=True, check=True
        )
        
        output_str = result.stdout
        json_start = output_str.find('{')
        if json_start == -1:
            print("[!] Could not find JSON output from pydeps.")
            return
            
        deps_data = json.loads(output_str[json_start:])
        
        in_degree = defaultdict(int)
        out_degree = defaultdict(int)
        
        # Collect all internal modules
        internal_modules = set()
        for module_name in deps_data.keys():
            if module_name == "__main__":
                continue
            if module_name.startswith(base_pkg):
                internal_modules.add(module_name)
        
        # Calculate degrees
        for module_name, data in deps_data.items():
            if module_name == "__main__":
                continue
                
            imports = data.get("imports", [])
            
            # For out-degree, we can count all imports this module makes
            if module_name in internal_modules:
                out_degree[module_name] = len(imports)
            
            # For in-degree, we only care about how many times an INTERNAL module is imported
            for imported_mod in imports:
                if imported_mod in internal_modules:
                    in_degree[imported_mod] += 1
                
        # Sort by in-degree (Hubs) and out-degree (Consumers)
        top_hubs = sorted(in_degree.items(), key=lambda x: x[1], reverse=True)
        
        # Find isolated code (In-Degree = 0) and Entry Points
        isolated_modules = []
        entry_points = []
        for mod in internal_modules:
            if in_degree[mod] == 0:
                if mod.startswith(f"{base_pkg}.demo") or mod.startswith(f"{base_pkg}.harness"):
                    entry_points.append(mod)
                else:
                    isolated_modules.append(mod)
        isolated_modules.sort()
        entry_points.sort()
        
        # Consumers (we can keep external imports in the count, or filter to only internal)
        # We will keep the out-degree as total imports to show complexity
        top_consumers = sorted(out_degree.items(), key=lambda x: x[1], reverse=True)
        
        # Generate Markdown
        md_content = f"# Codebase Topology Analysis for `{target_dir}`\n\n"
        md_content += "This report ranks the modules by their interconnectedness, helping to identify core 'Hubs' (highly imported), heavy 'Consumers' (importing many things), and isolated code.\n\n"
        
        md_content += f"## Top 20 Hubs (High In-Degree for `{base_pkg}`)\n"
        md_content += "Modules that provide core value and are depended upon by many others.\n\n"
        md_content += "| Module | In-Degree (Times Imported) |\n|---|---|\n"
        for mod, deg in top_hubs[:20]:
            md_content += f"| `{mod}` | {deg} |\n"
            
        md_content += "\n## Top 20 Consumers (High Out-Degree)\n"
        md_content += "Modules that orchestrate or combine many different components (more bug-prone).\n\n"
        md_content += "| Module | Out-Degree (Imports Made) |\n|---|---|\n"
        for mod, deg in top_consumers[:20]:
            md_content += f"| `{mod}` | {deg} |\n"

        md_content += "\n## Entry Points\n"
        md_content += "Known entry points and runner scripts that are not imported by other internal modules.\n\n"
        md_content += "| Module |\n|---|\n"
        for mod in entry_points:
            md_content += f"| `{mod}` |\n"
            
        md_content += "\n## Isolated / Dead Code (In-Degree = 0)\n"
        md_content += "Internal modules that are NEVER imported by anything else in the package. These are prime candidates for deletion or iceboxing (unless they are top-level entry point scripts).\n\n"
        md_content += "| Module |\n|---|\n"
        for mod in isolated_modules:
            md_content += f"| `{mod}` |\n"
        
        with open(output_md, "w") as f:
            f.write(md_content)
            
        print(f"[*] Markdown report generated at: {output_md}")
        
    except subprocess.CalledProcessError as e:
        print(f"[!] Error running pydeps: {e}")
        print(e.stderr)

if __name__ == "__main__":
    main()
