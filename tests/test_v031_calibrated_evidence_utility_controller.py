import pytest
import torch
from aegis.reliability.reliability_controller import (CalibratedEvidenceUtilityInterventionController,MonotoneEvidenceCalibrator,V031_ACTIVE_INTERVENTION_THRESHOLD,V031_MAX_INTERACTION_SUPPRESSION,V031_MAX_WEIGHT_SHIFT)
def scores():
    return (torch.tensor([[0.9],[0.2],[0.7]],dtype=torch.float32,requires_grad=True),torch.tensor([[0.4],[0.8],[0.7]],dtype=torch.float32,requires_grad=True),torch.tensor([[0.8],[0.3],[0.9]],dtype=torch.float32,requires_grad=True))
@pytest.mark.parametrize('mode',['calibration_only','utility_only','combined'])
def test_shapes_bounds(mode):
    o=CalibratedEvidenceUtilityInterventionController(mode=mode,utility_initialization_seed=42)(*scores())
    assert o.utility_features.shape==(3,7) and o.weights.shape==(3,2)
    for t in o.as_dict().values(): assert torch.isfinite(t).all()
    # Float32 tolerance applies only to numerical representation of the
    # mathematically frozen [0.10, 0.90] bounds. Exact g=0 reference identity
    # is verified separately and must not be replaced by clamping.
    tol=1e-7
    assert torch.all((o.text_weight>=.10-tol)&(o.text_weight<=.90+tol))
    assert torch.allclose(o.text_weight+o.vision_weight,torch.ones_like(o.text_weight),atol=tol,rtol=0.0)
    assert torch.all((o.interaction_multiplier>=.50)&(o.interaction_multiplier<=1.0))
def test_calibrator_monotone_identity():
    c=MonotoneEvidenceCalibrator(); x=torch.tensor([[.1],[.3],[.5],[.7],[.9]]) ; y=c(x)
    assert torch.all(y[1:]>y[:-1]); assert torch.allclose(x,y,atol=5e-5,rtol=5e-4)
@pytest.mark.parametrize('mode',['utility_only','combined'])
def test_no_intervention_initialization(mode):
    o=CalibratedEvidenceUtilityInterventionController(mode=mode,utility_initialization_seed=42)(*scores())
    assert torch.allclose(o.utility_probability,torch.full_like(o.utility_probability,.5)); assert torch.count_nonzero(o.intervention_gate)==0
    assert torch.count_nonzero(o.weight_intervention_magnitude)==0; assert torch.count_nonzero(o.interaction_suppression_magnitude)==0
def test_calibration_only_gate():
    o=CalibratedEvidenceUtilityInterventionController(mode='calibration_only')(*scores())
    assert torch.allclose(o.intervention_gate,((o.calibrated_risk-.25)/.50).clamp(0,1))
@pytest.mark.parametrize('mode',['calibration_only','utility_only','combined'])
def test_raw_inputs_detached(mode):
    o=CalibratedEvidenceUtilityInterventionController(mode=mode)(*scores())
    assert not o.q_text_raw.requires_grad and not o.q_vision_raw.requires_grad and not o.compatibility_raw.requires_grad
def test_calibrator_gradients():
    c=CalibratedEvidenceUtilityInterventionController(mode='combined'); o=c(*scores()); (o.q_text_calibrated.mean()+o.q_vision_calibrated.mean()+o.compatibility_calibrated.mean()).backward()
    assert c.text_calibrator.a_raw.grad is not None and c.vision_calibrator.a_raw.grad is not None and c.compatibility_calibrator.a_raw.grad is not None
def test_component_presence():
    c=CalibratedEvidenceUtilityInterventionController(mode='calibration_only'); u=CalibratedEvidenceUtilityInterventionController(mode='utility_only'); b=CalibratedEvidenceUtilityInterventionController(mode='combined')
    assert c.utility_selector is None and u.utility_selector is not None and b.utility_selector is not None
    assert c.text_calibrator is not None and u.text_calibrator is None and b.text_calibrator is not None
def test_seed_determinism():
    a=CalibratedEvidenceUtilityInterventionController(mode='combined',utility_initialization_seed=42); b=CalibratedEvidenceUtilityInterventionController(mode='combined',utility_initialization_seed=42)
    assert torch.equal(a.utility_selector[0].weight,b.utility_selector[0].weight)
def test_utility_target_formula_detached():
    r=torch.tensor([[2.,0.],[0.,2.]],requires_grad=True); c=torch.tensor([[3.,0.],[1.,2.]],requires_grad=True); y=torch.tensor([0,1],dtype=torch.long)
    o=CalibratedEvidenceUtilityInterventionController.utility_target_from_logits(r,c,y); rp=torch.softmax(r,1).gather(1,y[:,None]); cp=torch.softmax(c,1).gather(1,y[:,None]); d=(cp-rp).detach()
    assert torch.allclose(o['delta_u'],d); assert torch.allclose(o['u_target'],torch.sigmoid(d/.05)); assert all(not t.requires_grad for t in o.values())
def test_utility_target_rejects_invalid_class():
    with pytest.raises(ValueError): CalibratedEvidenceUtilityInterventionController.utility_target_from_logits(torch.zeros(2,2),torch.zeros(2,2),torch.tensor([0,2],dtype=torch.long))
def test_candidate_and_actual_bounds():
    o=CalibratedEvidenceUtilityInterventionController(mode='calibration_only')(*scores())
    assert torch.all((o.candidate_text_weight>=.10)&(o.candidate_text_weight<=.90)); assert torch.all((o.candidate_interaction_multiplier>=.50)&(o.candidate_interaction_multiplier<=1.0))
    assert torch.all(o.weight_intervention_magnitude<=V031_MAX_WEIGHT_SHIFT+1e-7); assert torch.all(o.interaction_suppression_magnitude<=V031_MAX_INTERACTION_SUPPRESSION+1e-7)
def test_active_threshold_and_metadata():
    c=CalibratedEvidenceUtilityInterventionController(mode='calibration_only'); o=c(*scores()); expected=(o.intervention_gate>=V031_ACTIVE_INTERVENTION_THRESHOLD).to(o.intervention_gate.dtype)
    assert torch.equal(o.active_intervention_indicator,expected); m=c.architecture_metadata(); assert m['official_test_accessed'] is False and m['calibration_loss_weight']==.25 and m['utility_loss_weight']==.50 and m['utility_target_temperature']==.05
def test_invalid_shape_nonfinite_rejected():
    c=CalibratedEvidenceUtilityInterventionController(mode='combined')
    with pytest.raises(ValueError): c(torch.ones(2),torch.ones(2,1),torch.ones(2,1))
    with pytest.raises(ValueError): c(torch.tensor([[float('nan')]]),torch.ones(1,1),torch.ones(1,1))
