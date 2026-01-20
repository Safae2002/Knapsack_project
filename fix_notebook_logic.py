
import json

notebook_path = "c:\\Users\\chaim\\OneDrive\\Desktop\\samba- graphe\\knapsack_project_python\\notebooks\\results_analysis.ipynb"

# Snippets to find and replace for the optimal rate logic
target_snippet = """    optimal_rate = df[df['Method'] == method]['Optimal'].mean() * 100"""
replacement_snippet = """    # Handle string boolean 'true'/'false' or native boolean
    opt_col = df[df['Method'] == method]['Optimal']
    if opt_col.dtype == 'object':
        optimal_rate = (opt_col.astype(str).str.lower() == 'true').mean() * 100
    else:
        optimal_rate = opt_col.mean() * 100"""

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if target_snippet in source:
             # Be careful with indentation and exact string match
             # It's better to process linewise or use exact replacement if we are sure of formatting
             # The previous script already updated keys, so 'optimal' became 'Optimal'
             # The target snippet above uses 'Optimal', which matches what is currently in the file 
             # (verified by view_file output in Step 328)
             
             new_source = source.replace(target_snippet, replacement_snippet)
             cell['source'] = [line + '\n' for line in new_source.split('\n')]
             
             # Cleanup trailing newlines validation
             if cell['source'][-1].strip() == "":
                 cell['source'][-1] = cell['source'][-1].rstrip() 
                 if cell['source'][-1] == "":
                     cell['source'].pop()

# Save
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook logic updated.")
