import os, glob, re

paths = glob.glob('**/*.yaml', recursive=True) + glob.glob('**/*.py', recursive=True)
paths = [p for p in paths if 'mlruns' not in p]

updated = 0
for path in paths:
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        c = f.read()
    orig = c
    c = re.sub(r'from ml\.', 'from training.', c)
    c = re.sub(r'import ml\.', 'import training.', c)
    c = c.replace('"training/', '"training/')
    c = c.replace("'training/", "'training/")
    c = c.replace(' training/', ' training/')
    c = c.replace(' training\\\\', ' training\\\\')
    
    if orig != c:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(c)
        print(f'Updated {path}')
        updated += 1

print(f"Total files updated: {updated}")
