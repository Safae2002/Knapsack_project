import json

notebook_path = "c:\\Users\\chaim\\OneDrive\\Desktop\\samba- graphe\\knapsack_project_python\\notebooks\\results_analysis.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

print("Searching for 'method' references in notebook cells:")
print("=" * 60)

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if "'method'" in source or '"method"' in source:
            print(f"\nCell {i} contains 'method':")
            print(source[:200])
            print("...")
