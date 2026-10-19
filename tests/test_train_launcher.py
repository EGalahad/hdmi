"""Check the download gate without starting CUDA workers or accessing the Hub."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


@pytest.mark.parametrize('prepare_exit', [0, 7])
def test_cache_preparation_gates_distributed_launch(tmp_path, prepare_exit):
    root = tmp_path / 'active-adaptation'
    scripts = root / 'projects/hdmi/scripts'
    scripts.mkdir(parents=True)
    shutil.copyfile(Path(__file__).parents[1] / 'scripts/train.sh', scripts / 'train.sh')
    (root / 'scripts').mkdir()
    (root / 'scripts/launch_ddp.sh').write_text(
        '#!/usr/bin/env bash\nexec uv fake-ddp "$@"\n'
    )
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    uv = bin_dir / 'uv'
    uv.write_text(
        f'#!{sys.executable}\n'
        'import json, os, sys\n'
        'with open(os.environ["CALL_LOG"], "a") as f:\n'
        '    f.write(json.dumps({"args": sys.argv[1:], "offline": os.getenv("HF_HUB_OFFLINE"), "cwd": os.getcwd()}) + "\\n")\n'
        'sys.exit(0 if sys.argv[1] == "fake-ddp" else int(os.environ["PREPARE_EXIT"]))\n'
    )
    uv.chmod(0o755)
    log = tmp_path / 'calls.jsonl'
    env = dict(os.environ, PATH=f'{bin_dir}:{os.environ["PATH"]}',
               CALL_LOG=str(log), PREPARE_EXIT=str(prepare_exit),
               AA_UV_PROJECT='venv/custom env')
    env.pop('HF_HUB_OFFLINE', None)
    overrides = ['task=weave-contact-full', '+exp=hdmi/weave', 'wandb.name=run with spaces']
    proc = subprocess.run(['bash', str(scripts / 'train.sh'), '0,2', *overrides],
                          cwd=tmp_path, env=env, capture_output=True, text=True)
    assert proc.returncode == prepare_exit, proc.stderr
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert calls[0]['offline'] is None
    assert calls[0]['cwd'] == str(root)
    assert all(arg in calls[0]['args'] for arg in overrides)
    if prepare_exit:
        assert len(calls) == 1
    else:
        assert len(calls) == 2
        assert calls[1]['offline'] == '1'
        assert calls[1]['args'] == ['fake-ddp', '0,2', 'projects/mimic-lite/scripts/train.py', 'venv/custom env', *overrides]
