import torch
import pytest

from aegis.alignment import CrossModalAlignmentModel
from aegis.classification import HierarchicalInformationIntegrityClassifier
from aegis.training import BinaryIntegrityBatch
from scripts.run_fakeddit_ablation import (
    FakedditAblationTrainer,
    V031_CALIBRATION_LOSS_WEIGHT,
    V031_UTILITY_LOSS_WEIGHT,
    V031_UTILITY_TARGET_TEMPERATURE,
    is_v031_calibrated_intervention_architecture,
    v031_calibrated_intervention_mode,
)

ARCHS={
 "quality_compatibility_calibrated_intervention_calibration":"calibration_only",
 "quality_compatibility_calibrated_intervention_utility":"utility_only",
 "quality_compatibility_calibrated_intervention":"combined",
}

def make(mode):
    torch.manual_seed(42)
    m=CrossModalAlignmentModel(
        text_dim=8,vision_dim=6,shared_dim=4,dropout=0.0,temperature=0.07,
        quality_compatibility_calibrated_intervention=mode,
        v031_utility_initialization_seed=42,
        quality_hidden_dims=(8,4),quality_dropout=0.0,
        compatibility_hidden_dims=(8,4),compatibility_dropout=0.0,
    )
    c=HierarchicalInformationIntegrityClassifier(
        input_dim=4,hidden_dim=4,dropout=0.0
    )
    opt=torch.optim.AdamW(
        list(m.parameters())+list(c.parameters()),lr=1e-3,weight_decay=1e-4
    )
    arch={v:k for k,v in ARCHS.items()}[mode]
    t=FakedditAblationTrainer(
        alignment_model=m,classification_model=c,optimizer=opt,
        device=torch.device("cpu"),alignment_loss_weight=0.5,
        classification_loss_weight=1.0,gradient_clip_norm=1.0,
        mode="multimodal",fusion_architecture=arch,
    )
    return m,c,opt,t

def data():
    torch.manual_seed(7)
    b=BinaryIntegrityBatch(
        text_embeddings=torch.randn(4,8),
        vision_embeddings=torch.randn(4,6),
        integrity_targets=torch.tensor([0,1,0,1],dtype=torch.long),
    )
    qt=torch.tensor([[1.0],[0.7],[0.4],[0.2]])
    qv=torch.tensor([[0.8],[0.5],[0.9],[0.3]])
    comp=torch.tensor([[1.0],[0.0],[1.0],[0.0]])
    return b,(qt,qv),comp

@pytest.mark.parametrize("arch,mode",ARCHS.items())
def test_routing(arch,mode):
    assert is_v031_calibrated_intervention_architecture(arch)
    assert v031_calibrated_intervention_mode(arch)==mode
    m,_,_,t=make(mode)
    assert t.fusion_architecture==arch
    assert m.calibrated_utility_controller.mode==mode

def test_frozen_constants():
    assert V031_CALIBRATION_LOSS_WEIGHT==0.25
    assert V031_UTILITY_LOSS_WEIGHT==0.50
    assert V031_UTILITY_TARGET_TEMPERATURE==0.05

@pytest.mark.parametrize("mode",["utility_only","combined"])
def test_utility_target_detached(mode):
    _,_,_,t=make(mode); b,q,c=data()
    o=t.forward_batch(
        b,quality_targets=q,quality_loss_weight=1.0,
        compatibility_targets=c,compatibility_loss_weight=1.0,
        calibration_loss_weight=0.25 if mode=="combined" else 0.0,
        utility_loss_weight=0.50,transition_loss_weight=0.0,
    )
    assert o["utility_target_outputs"] is not None
    for x in o["utility_target_outputs"].values():
        assert not x.requires_grad

@pytest.mark.parametrize("arch,mode",ARCHS.items())
def test_real_one_batch_training_path(arch,mode):
    m,_,opt,t=make(mode); b,q,c=data()
    ctrl=m.calibrated_utility_controller
    before={n:p.detach().clone() for n,p in ctrl.named_parameters()}
    out=t.train_step(
        b,quality_targets=q,quality_loss_weight=1.0,
        compatibility_targets=c,compatibility_loss_weight=1.0,
        calibration_loss_weight=0.25 if mode in {"calibration_only","combined"} else 0.0,
        utility_loss_weight=0.50 if mode in {"utility_only","combined"} else 0.0,
        transition_reference_batch=None,transition_loss_weight=0.0,
    )
    assert out["transition_loss"]==0.0
    assert torch.isfinite(torch.tensor(out["loss"]))
    assert (out["calibration_loss"]>0.0)==(mode in {"calibration_only","combined"})
    assert (out["utility_loss"]>0.0)==(mode in {"utility_only","combined"})
    after=dict(ctrl.named_parameters())
    assert before
    assert any(not torch.equal(before[n],after[n].detach()) for n in before)
    ids={id(p) for g in opt.param_groups for p in g["params"]}
    assert all(id(p) in ids for p in ctrl.parameters())
