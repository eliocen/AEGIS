from pathlib import Path
from torch import nn
import timm
from safetensors.torch import load_file
MODEL_NAME="vit_small_patch16_224.dino"
FEATURE_DIM=384
class AegisVisionModel(nn.Module):
    def __init__(self,encoder,reliability=False):
        super().__init__(); self.encoder=encoder
        self.authenticity_head=nn.Linear(FEATURE_DIM,1)
        self.reliability_head=nn.Sequential(nn.Linear(FEATURE_DIM,128),nn.GELU(),nn.Dropout(0.1),nn.Linear(128,1)) if reliability else None
    def forward(self,x):
        z=self.encoder.forward_features(x)
        if z.ndim==3: z=z[:,0]
        return {"authenticity_logit":self.authenticity_head(z).squeeze(-1),"reliability_logit":None if self.reliability_head is None else self.reliability_head(z.detach()).squeeze(-1)}
def _load_encoder(weight_path):
    enc=timm.create_model(MODEL_NAME,pretrained=False,num_classes=0)
    state=load_file(str(Path(weight_path)))
    missing,unexpected=enc.load_state_dict(state,strict=False)
    bad_missing=[x for x in missing if not x.startswith("head")]
    bad_unexpected=[x for x in unexpected if not x.startswith("head")]
    if bad_missing or bad_unexpected: raise RuntimeError(f"weight incompatibility missing={bad_missing} unexpected={bad_unexpected}")
    return enc
def configure_trainability(model,variant):
    for p in model.parameters(): p.requires_grad=False
    for p in model.authenticity_head.parameters(): p.requires_grad=True
    if variant in ("B1","A1"):
        for block in model.encoder.blocks[-2:]:
            for p in block.parameters(): p.requires_grad=True
        for p in model.encoder.norm.parameters(): p.requires_grad=True
    if variant=="A1":
        for p in model.reliability_head.parameters(): p.requires_grad=True
    return model
def build_model(variant,weight_path):
    if variant not in ("B0","B1","A1"): raise ValueError(variant)
    return configure_trainability(AegisVisionModel(_load_encoder(weight_path),variant=="A1"),variant)
