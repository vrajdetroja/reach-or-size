import json
import math

from analysis.extension import diff_of_gains, extra_seed_candidates, joint_reading, load_cell, verdict


def test_verdict_follows_the_preregistered_rules():
    assert verdict(0.021, 0.010, 0.0056) == 'positive'      # X - 2 SE = +0.001
    assert verdict(0.019, 0.010, 0.0056) == 'inconclusive'  # neither bound
    assert verdict(-0.012, 0.010, 0.0056) == 'absent'       # X + 1.645 SE = 0.00445 < m
    assert verdict(0.0, 0.004, 0.0056) == 'inconclusive'    # 0.00658 is not below m


def test_diff_of_gains_uses_all_four_cells():
    c = lambda mean, var: dict(mean=mean, var=var, n=3)
    x, se = diff_of_gains(c(0.70, 3e-4), c(0.67, 3e-4), c(0.717, 3e-4), c(0.706, 3e-4))
    assert math.isclose(x, 0.019, abs_tol=1e-12) and math.isclose(se, math.sqrt(4e-4), rel_tol=1e-9)


def test_joint_reading_rows():
    assert joint_reading('positive', 'inconclusive', 'absent', False) == ['Evidence for H-size']
    assert joint_reading('absent', 'positive', 'inconclusive', False)[0].startswith('Evidence for H-reach')
    assert joint_reading('inconclusive', 'inconclusive', 'absent', True)[0].startswith('Evidence for H-local')
    assert joint_reading('inconclusive', 'inconclusive', 'absent', False)[0].startswith('Inconclusive')
    assert joint_reading('positive', 'positive', 'inconclusive', False) == ['RWSE does more than one job; we quantify each']
    # B or C positive without A absent is not the H-reach row.
    assert joint_reading('inconclusive', 'positive', 'inconclusive', False)[0].startswith('Inconclusive')


def test_extra_seed_candidates_in_priority_order():
    arms = dict(A=dict(x=0.015, se=0.01), B=dict(x=0.012, se=0.01), C=dict(x=0.015, se=0.01))
    assert extra_seed_candidates(arms) == ['B', 'A']  # C is never extended
    assert extra_seed_candidates(dict(A=dict(x=0.025, se=0.01), B=dict(x=0.005, se=0.01))) == []


def test_load_cell_skips_tagged_runs(tmp_path):
    for run, ap in (('cfg_s0', 0.7), ('cfg_s1', 0.71), ('cfg_s0_smoke', 0.5), ('cfg_nope_s0', 0.6)):
        (tmp_path / run).mkdir()
        (tmp_path / run / 'summary.json').write_text(json.dumps(dict(test_at_best_val=ap)))
    assert sorted(load_cell(str(tmp_path), 'cfg')) == [0.7, 0.71]
