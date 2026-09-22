import torch
from aegis.vision.losses import reliability_target,compute_losses
from aegis.vision.config import TrainingConfig
def test_reliability_target_is_detached():
    z=torch.tensor([2.,-2.],requires_grad=True); y=torch.tensor([1.,1.]); t=reliability_target(z,y)
    assert not t.requires_grad and t.tolist()==[1.,0.]
def test_reliability_loss_does_not_flow_via_target():
    a=torch.tensor([.2,-.3],requires_grad=True); r=torch.tensor([.4,.1],requires_grad=True); y=torch.tensor([1.,0.])
    g=torch.autograd.grad(compute_losses({"authenticity_logit":a,"reliability_logit":r},y)["reliability"],a,allow_unused=True)
    assert g[0] is None
def test_frozen_config():
    c=TrainingConfig(); assert c.physical_batch*c.grad_accumulation==64
    assert c.seeds==(42,43,44) and c.reliability_weight==0.25
