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
def test_a1_reliability_representation_is_detached_from_shared_encoder_source():
    """Frozen Step3B: reliability loss trains the reliability head only by default."""
    import inspect
    from aegis.vision.model import AegisVisionModel
    src = inspect.getsource(AegisVisionModel.forward)
    assert "self.reliability_head(z.detach())" in src
def test_t1_manifest_semantic_label_encoding(tmp_path):
    import csv
    from aegis.vision.data import UsablePopulationDataset
    manifest = tmp_path / "m.csv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["relative_path","native_split","mapped_label"])
        w.writeheader()
        w.writerow({"relative_path":"a.jpg","native_split":"train","mapped_label":"AUTHENTIC"})
        w.writerow({"relative_path":"b.png","native_split":"train","mapped_label":"MANIPULATED_AI_GENERATED"})
        w.writerow({"relative_path":"c.jpg","native_split":"val","mapped_label":"AUTHENTIC"})
    ds = UsablePopulationDataset(manifest, tmp_path, "train", lambda x:x)
    assert ds.rows == [("a.jpg",0.0),("b.png",1.0)]