import json
import os

notebook_path = '/home/naresh/Downloads/voc-training/voc_biometric copy.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Update metadata
nb['metadata']['kernelspec'] = {
    "display_name": "Python 3.13 (VOC Training)",
    "language": "python",
    "name": "voc_biometric"
}

# Fix Colab code in cells
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        new_source = []
        changed = False
        i = 0
        while i < len(source):
            line = source[i]
            if 'from google.colab import files' in line:
                new_source.append('try:\n')
                new_source.append('    from google.colab import files\n')
                new_source.append('    COLAB = True\n')
                new_source.append('except ImportError:\n')
                new_source.append('    COLAB = False\n')
                changed = True
            elif 'files.download(' in line:
                indent = line[:line.find('files.download(')]
                new_source.append(f'{indent}if COLAB:\n')
                new_source.append(f'{indent}    {line.strip()}\n')
                new_source.append(f'{indent}    print("✅ Downloaded via Colab.")\n')
                new_source.append(f'{indent}else:\n')
                new_source.append(f'{indent}    print("✅ Local environment: files saved to disk.")\n')
                changed = True
            else:
                new_source.append(line)
            i += 1
        if changed:
            cell['source'] = new_source

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Successfully updated notebook metadata and fixed Colab dependencies.")
