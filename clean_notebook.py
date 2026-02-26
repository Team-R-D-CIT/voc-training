import json
import os
import re

notebook_path = '/home/naresh/Downloads/voc-training/voc_biometric copy.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Update metadata
nb['metadata']['kernelspec'] = {
    "display_name": "Python 3.13 (VOC Training)",
    "language": "python",
    "name": "voc_biometric"
}

# Clean cells
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        new_source = []
        for line in source:
            # Remove colab imports
            if 'from google.colab import files' in line:
                continue
            # Replace download with print
            if 'files.download(' in line:
                filename_match = re.search(r"files\.download\(['\"](.+?)['\"]\)", line)
                if filename_match:
                    filename = filename_match.group(1)
                    indent = line[:line.find('files.download(')]
                    new_source.append(f"{indent}print(f'✅ File saved locally: {{os.path.abspath(\"{filename}\")}}')\n")
                else:
                    new_source.append(f"# {line}")
                continue
            # Remove other colab calls if any
            if 'google.colab' in line:
                new_source.append(f"# {line}")
                continue
            
            new_source.append(line)
        cell['source'] = new_source

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook cleaned: Google Colab code removed and local kernel set.")
