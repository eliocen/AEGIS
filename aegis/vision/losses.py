import torch
import torch.nn.functional as F
def reliability_target(authenticity_logit,label):
    with torch.no_grad():
        pred=(authenticity_logit.detach()>=0).to(label.dtype)
        return (pred==label).to(authenticity_logit.dtype)
def compute_losses(output,label,reliability_weight=0.25):
    label=label.to(output["authenticity_logit"].dtype)
    auth=F.binary_cross_entropy_with_logits(output["authenticity_logit"],label)
    rel_logit=output.get("reliability_logit")
    if rel_logit is None:return {"total":auth,"authenticity":auth,"reliability":None}
    rel=F.binary_cross_entropy_with_logits(rel_logit,reliability_target(output["authenticity_logit"],label))
    return {"total":auth+reliability_weight*rel,"authenticity":auth,"reliability":rel}
