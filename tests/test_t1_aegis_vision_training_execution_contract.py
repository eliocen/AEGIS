import math
import numpy as np
import pytest
import torch
from scripts import run_t1_aegis_vision_training as R

def test_lr_schedule_boundaries():
    assert R.lr_factor(1,10,100)==pytest.approx(0.1)
    assert R.lr_factor(10,10,100)==pytest.approx(1.0)
    assert R.lr_factor(100,10,100)==pytest.approx(0.0)

def test_checkpoint_lexicographic_and_reliability_irrelevant():
    a={"macro_f1":0.8,"bce":0.3,"reliability":{"brier":0.99}}
    b={"macro_f1":0.8,"bce":0.2,"reliability":{"brier":0.01}}
    assert R.checkpoint_key(b,9)<R.checkpoint_key(a,1)
    assert R.checkpoint_key(b,2)<R.checkpoint_key(b,3)

def test_metrics_known_case_and_ties():
    m=R.authenticity_metrics([0,0,1,1],[-2,1,-1,2],4.0)
    assert m["macro_f1"]==pytest.approx(0.5)
    assert m["balanced_accuracy"]==pytest.approx(0.5)
    assert m["bce"]==pytest.approx(1.0)
    assert R.binary_auroc([0,1],[0.5,0.5])==pytest.approx(0.5)

def test_parameter_groups_use_frozen_rates():
    class M(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.encoder=torch.nn.Linear(2,2); self.authenticity_head=torch.nn.Linear(2,1)
    m=M(); c=R.TrainingConfig(); g=R.parameter_groups(m,c)
    assert [x["name"] for x in g]==["backbone","heads"]
    assert [x["base_lr"] for x in g]==[c.backbone_lr,c.head_lr]

def test_dry_snapshot_zero_steps_and_locked():
    d=R.dry_snapshot()
    assert d["formal_runs"]==9 and d["effective_batch"]==64
    assert d["optimizer_steps"]==0 and d["training_authorized"] is False
    assert d["sample_weighted_accumulation"] is True and d["drop_last"] is False

def test_authorization_rejects_wrong_runner(tmp_path):
    p=tmp_path/"a.json"
    p.write_text('{"schema":"AEGIS_T1_STEP8_TRAINING_AUTHORIZATION_V1","training_authorized":true,"runner_commit":"wrong","step7d_commit":"3cc77b075793568a93784be21eb8ae56225c079f"}')
    with pytest.raises(RuntimeError):R.validate_authorization(p,"right")
def test_reliability_degenerate_metrics_are_json_safe():
    m=R.reliability_metrics([0,1],[-1,1],[2,2])
    assert m["correctness_auroc"] is None
    assert m["mean_incorrect"] is None
    import json
    json.dumps(m,allow_nan=False)

def test_authorization_accepts_frozen_runner_blob_identity(tmp_path):
    p=tmp_path/"a.json"
    p.write_text('{"schema":"AEGIS_T1_STEP8_TRAINING_AUTHORIZATION_V1","training_authorized":true,"runner_commit":"runner-blob-id","step7d_commit":"3cc77b075793568a93784be21eb8ae56225c079f"}')
    assert R.validate_authorization(p,"runner-blob-id")["training_authorized"] is True
