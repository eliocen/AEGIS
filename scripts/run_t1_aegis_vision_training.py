import argparse, json, math, random
from pathlib import Path
import numpy as np
import torch
from torch.optim import AdamW
from aegis.vision.config import TrainingConfig

VARIANTS=("B0","B1","A1")
SEEDS=(42,43,44)

def seed_everything(seed:int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def parameter_groups(model,cfg):
    backbone=[];heads=[]
    for n,p in model.named_parameters():
        if not p.requires_grad: continue
        (backbone if n.startswith("encoder.") else heads).append(p)
    groups=[]
    if backbone: groups.append({"params":backbone,"lr":cfg.backbone_lr})
    if heads: groups.append({"params":heads,"lr":cfg.head_lr})
    return groups

def build_optimizer(model,cfg):
    return AdamW(parameter_groups(model,cfg),weight_decay=cfg.weight_decay)

def checkpoint_key(m):
    return (-float(m["macro_f1"]),float(m["auth_bce"]),int(m["epoch"]))

def better_checkpoint(candidate,best):
    return best is None or checkpoint_key(candidate)<checkpoint_key(best)

def planned_run_roots(root=Path("artifacts/post_v1/t1_aegis_vision/training")):
    return [root/v/f"seed_{s}" for v in VARIANTS for s in SEEDS]

def protocol_snapshot():
    c=TrainingConfig()
    return {"variants":list(VARIANTS),"seeds":list(SEEDS),"formal_runs":9,
      "optimizer":"AdamW","backbone_lr":c.backbone_lr,"head_lr":c.head_lr,
      "weight_decay":c.weight_decay,"physical_batch":c.physical_batch,
      "gradient_accumulation":c.grad_accumulation,
      "effective_batch":c.physical_batch*c.grad_accumulation,
      "gradient_clip_norm":c.grad_clip,"reliability_weight":c.reliability_weight,
      "checkpoint_order":["highest validation authenticity Macro-F1","lowest validation authenticity BCE","earliest epoch"],
      "optimizer_steps":0,"training_authorized":False}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dry-run",action="store_true",required=True)
    args=ap.parse_args()
    if not args.dry_run: raise SystemExit("training execution is not authorized")
    print(json.dumps(protocol_snapshot(),indent=2))
if __name__=="__main__": main()
