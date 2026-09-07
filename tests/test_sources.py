from pathlib import Path
import hashlib
import json


def test_vendored_source_hashes():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root/'SOURCE_MANIFEST.json').read_text())
    for record in manifest['files']:
        assert hashlib.sha256((root/record['target']).read_bytes()).hexdigest() == record['adapted_sha256']


def test_shell_script_lf_and_all_configs_present():
    root = Path(__file__).resolve().parents[1]
    script = (root/'scripts/install_ubuntu.sh').read_bytes()
    assert b'\r' not in script
    assert script.startswith(b'#!/usr/bin/env bash\n')
    assert len(list((root/'configs/main').glob('*.yaml'))) == 36
    assert len(list((root/'configs/ablation').glob('*.yaml'))) == 10
