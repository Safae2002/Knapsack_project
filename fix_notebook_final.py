import json
import re

notebook_path = "c:\\Users\\chaim\\OneDrive\\Desktop\\samba- graphe\\knapsack_project_python\\notebooks\\results_analysis.ipynb"

# Column name mappings
replacements = {
    r"\['method'\]": "['Method']",
    r'\["method"\]': '["Method"]',
    r"\['value'\]": "['Value']",
    r'\["value"\]': '["Value"]',
    r"\['time_ms'\]": "['Time(ms)']",
    r'\["time_ms"\]': '["Time(ms)"]',
    r"\['optimal'\]": "['Optimal']",
    r'\["optimal"\]': '["Optimal"]',
    r"\['instance'\]": "['Instance']",
    r'\["instance"\]': '["Instance"]',
    r"\['difficulty'\]": "['Difficulty']",
    r'\["difficulty"\]': '["Difficulty"]',
    r"\['gap_percent'\]": "['Gap(%)']",
    r'\["gap_percent"\]': '["Gap(%)"]',
}

# Also handle dictionary key definitions
dict_replacements = {
    r"'method':": "'Method':",
    r'"method":': '"Method":',
    r"'value':": "'Value':",
    r'"value":': '"Value":',
    r"'time_ms':": "'Time(ms)':",
    r'"time_ms":': '"Time(ms)":',
    r"'optimal':": "'Optimal':",
    r'"optimal":': '"Optimal":',
    r"'instance':": "'Instance':",
    r'"instance":': '"Instance":',
    r"'difficulty':": "'Difficulty':",
    r'"difficulty":': '"Difficulty":',
}

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

changes_made = 0

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source_lines = cell['source']
        new_source_lines = []
        
        for line in source_lines:
            new_line = line
            
            # Apply regex replacements for column access
            for pattern, replacement in replacements.items():
                new_line = re.sub(pattern, replacement, new_line)
            
            # Apply dictionary key replacements
            for old, new in dict_replacements.items():
                new_line = new_line.replace(old, new)
            
            if new_line != line:
                changes_made += 1
                
            new_source_lines.append(new_line)
        
        cell['source'] = new_source_lines

# Save
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Notebook updated. Made {changes_made} changes.")
print("\n⚠️  IMPORTANT: You MUST restart your Jupyter kernel and reload the notebook!")
print("   1. In Jupyter, click 'Kernel' -> 'Restart'")
print("   2. Then reload the page (F5 or Ctrl+R)")
print("   3. Re-run all cells")
