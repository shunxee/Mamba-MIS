"""Create a portable source archive without environments, datasets, results or clones."""
from pathlib import Path
import zipfile

root = Path(__file__).resolve().parents[1]
target = root.parent / 'Mamba-MIS.zip'
allowed_dirs = {'src','configs','docs','scripts','tests','tools'}
allowed_files = {'README.md','pyproject.toml','requirements.txt','SOURCE_MANIFEST.json','.gitignore'}
with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as archive:
    candidates = [root/name for name in allowed_files]
    candidates += [p for directory in allowed_dirs for p in (root/directory).rglob('*')]
    for item in sorted(candidates):
        rel = item.relative_to(root)
        if not item.is_file() or '__pycache__' in rel.parts or any(part.endswith('.egg-info') for part in rel.parts):
            continue
        if rel.parts[0] not in allowed_dirs and rel.as_posix() not in allowed_files:
            continue
        archive.write(item,Path(root.name)/rel)
print(target)
