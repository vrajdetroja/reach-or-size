import json

import pytest
import yaml

from src.train import load_subset


def test_subset_files_are_nested_and_hash_checked(tmp_path):
    a = load_subset('configs/subsets/peptides-func_train_frac25_seed0.json', 10873)
    b = load_subset('configs/subsets/peptides-func_train_frac50_seed0.json', 10873)
    assert (len(a), len(b)) == (2718, 5436) and set(a) <= set(b)
    spec = json.load(open('configs/subsets/peptides-func_train_frac25_seed0.json'))
    spec['indices'][0] += 1
    bad = tmp_path / 'bad.json'
    bad.write_text(json.dumps(spec))
    with pytest.raises(ValueError):
        load_subset(str(bad), 10873)
    with pytest.raises(ValueError):
        load_subset('configs/subsets/peptides-func_train_frac25_seed0.json', 10000)


def test_arm_configs_differ_from_base_only_where_preregistered():
    base = yaml.safe_load(open('configs/runs/gcnplus_peptides-func.yaml'))
    allowed = {'armA': {'pe_steps', 'subset_file', 'steps_per_epoch'},
               'armB': {'pe_steps', 'residual', 'layers', 'hidden'},
               'armC': {'pe_steps'}}
    for arm, keys in allowed.items():
        for path in __import__('glob').glob(f'configs/extension/*{arm}*.yaml'):
            cfg = yaml.safe_load(open(path))
            changed = {k for sec in ('model', 'train') for k in set(cfg[sec]) | set(base[sec])
                       if cfg[sec].get(k) != base[sec].get(k)}
            assert changed <= keys, (path, changed)
            assert cfg['train']['epochs'] == 300 and cfg['dataset'] == 'peptides-func'
