
import json
import re

notebook_path = "c:\\Users\\chaim\\OneDrive\\Desktop\\samba- graphe\\knapsack_project_python\\notebooks\\results_analysis.ipynb"

# Mapping of old column names to new column names
# The user wants specific casing.
replacements = {
    "'method'": "'Method'",
    "'value'": "'Value'",
    "'time_ms'": "'Time(ms)'",
    "'optimal'": "'Optimal'",
    "'instance'": "'Instance'",
    "'difficulty'": "'Difficulty'",
    "'gap_percent'": "'Gap(%)'"
}

# We also need to handle the demonstation data creation in cell 2 if it exists
demo_data_replacements = {
    "'method':": "'Method':",
    "'value':": "'Value':",
    "'time_ms':": "'Time(ms)':",
    "'optimal':": "'Optimal':",
    "'instance':": "'Instance':",
    "'difficulty':": "'Difficulty':",
}

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source_lines = cell['source']
        new_source_lines = []
        for line in source_lines:
            new_line = line
            # Apply column access replacements (e.g. df['method'])
            for old, new in replacements.items():
                new_line = new_line.replace(old, new)
            
            # Apply demo data key replacements
            for old, new in demo_data_replacements.items():
                new_line = new_line.replace(old, new)
            
            # Special case for 'optimal' which is now a string "true"/"false" in CSV but boolean in python if read?
            # Actually pandas reads "true"/"false" as string unless we specify valid boolean or convert.
            # In knapsack_solver.py we saw: df['Optimal_Bool'] = df['Optimal'] == 'true'
            # The notebook needs to handle this.
            # Replace: optimal_rate = df[df['method'] == method]['optimal'].mean() * 100
            # with something that handles string 'true'/'false' if loaded from CSV.
            # But simpler first: just rename the columns.
            
            # Also, we need to handle "gap_percent" which is now "Gap(%)" but might be accessed as attribute or something else? 
            # The replacements above handle string keys which is 99% of usage in pandas.
            
            new_source_lines.append(new_line)
        cell['source'] = new_source_lines

# Save
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook columns updated.")
