from pathlib import Path
import torch
from torch import nn
from scripts.run_t1_aegis_vision_training import (
    VARIANTS,SEEDS,better_checkpoint,parameter_groups,planned_run_roots,protocol_snapshot
)
from aegis.vision.config import TrainingConfig

class Dummy(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.Linear(2,2)
        self.authenticity_head=nn.Linear(2,1)
        self.reliability_head=nn.Linear(2,1)

def test_training_matrix_and_roots_are_exact():
    assert VARIANTS==("B0","B1","A1") and SEEDS==(42,43,44)
    roots=planned_run_roots(Path("ROOT"))
    assert len(roots)==9 and len(set(map(str,roots)))==9

def test_checkpoint_selection_is_lexicographic_and_reliability_independent():
    best={"macro_f1":.8,"auth_bce":.4,"epoch":3,"rel_brier":.01}
    assert better_checkpoint({"macro_f1":.81,"auth_bce":.9,"epoch":9,"rel_brier":.99},best)
    assert better_checkpoint({"macro_f1":.8,"auth_bce":.39,"epoch":9,"rel_brier":.99},best)
    assert better_checkpoint({"macro_f1":.8,"auth_bce":.4,"epoch":2,"rel_brier":.99},best)
    assert not better_checkpoint({"macro_f1":.8,"auth_bce":.4,"epoch":4,"rel_brier":0.0},best)

def test_optimizer_groups_preserve_frozen_learning_rates():
    m=Dummy();c=TrainingConfig();groups=parameter_groups(m,c)
    assert [g["lr"] for g in groups]==[c.backbone_lr,c.head_lr]

def test_dry_snapshot_authorizes_zero_optimizer_steps():
    s=protocol_snapshot()
    assert s["formal_runs"]==9 and s["effective_batch"]==64
    assert s["optimizer_steps"]==0 and s["training_authorized"] is False
    assert s["reliability_weight"]==0.25
