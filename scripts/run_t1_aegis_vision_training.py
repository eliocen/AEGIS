from __future__ import annotations
import argparse, csv, hashlib, json, math, os, random, subprocess
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from aegis.vision.config import TrainingConfig
from aegis.vision.data import UsablePopulationDataset, build_transforms
from aegis.vision.losses import compute_losses
from aegis.vision.model import build_model

VARIANTS=("B0","B1","A1")
SEEDS=(42,43,44)
MANIFEST_SHA="09CF00DB0B3217677B615DA8E38B706A270B8EFA4FBF39F8A6EA0747C6C7161F"
WEIGHT_SHA="8D4795B8E4E327C6D841E15A2E3E6D88C45346256A43F8CDAE8C26D92C806F7A"
STEP7D_COMMIT="3cc77b075793568a93784be21eb8ae56225c079f"
DEFAULT_MANIFEST=Path("data/manifests/genimage/t1_step2d14_sd14_usable_population/usable_population_manifest.csv")
DEFAULT_RAW_ROOT=Path("data/raw/genimage/stable_diffusion_v_1_4/extracted/imagenet_ai_0419_sdv4")
DEFAULT_WEIGHT=Path("data/models/timm/vit_small_patch16_224.dino/model.safetensors")
DEFAULT_ARTIFACT_ROOT=Path("artifacts/post_v1/t1_aegis_vision/training")

def sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""): h.update(b)
    return h.hexdigest().upper()

def git_head():
    return subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()

def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def seed_worker(_worker_id):
    s=torch.initial_seed()%2**32
    random.seed(s); np.random.seed(s)

def parameter_groups(model,cfg):
    backbone=[]; heads=[]
    for n,p in model.named_parameters():
        if not p.requires_grad: continue
        (backbone if n.startswith("encoder.") else heads).append(p)
    groups=[]
    if backbone: groups.append({"params":backbone,"lr":cfg.backbone_lr,"base_lr":cfg.backbone_lr,"name":"backbone"})
    if heads: groups.append({"params":heads,"lr":cfg.head_lr,"base_lr":cfg.head_lr,"name":"heads"})
    return groups

def lr_factor(update_index,warmup_updates,total_updates):
    if not (1<=update_index<=total_updates): raise ValueError("update index out of range")
    if warmup_updates>0 and update_index<=warmup_updates:
        return update_index/warmup_updates
    remain=total_updates-warmup_updates
    if remain<=0:return 1.0
    progress=(update_index-warmup_updates)/remain
    return 0.5*(1.0+math.cos(math.pi*progress))

def apply_lr(optimizer,update_index,warmup_updates,total_updates):
    factor=lr_factor(update_index,warmup_updates,total_updates)
    for g in optimizer.param_groups:g["lr"]=g["base_lr"]*factor
    return factor

def _average_ranks(x):
    order=np.argsort(x,kind="mergesort"); ranks=np.empty(len(x),dtype=np.float64)
    i=0
    while i<len(x):
        j=i+1
        while j<len(x) and x[order[j]]==x[order[i]]:j+=1
        ranks[order[i:j]]=(i+j-1)/2.0+1.0; i=j
    return ranks

def binary_auroc(y,score):
    y=np.asarray(y,dtype=np.int64); score=np.asarray(score,dtype=np.float64)
    n1=int((y==1).sum()); n0=int((y==0).sum())
    if n1==0 or n0==0:return float("nan")
    r=_average_ranks(score)
    return float((r[y==1].sum()-n1*(n1+1)/2.0)/(n1*n0))

def authenticity_metrics(labels,logits,bce_sum):
    y=np.asarray(labels,dtype=np.int64); z=np.asarray(logits,dtype=np.float64); p=(z>=0).astype(np.int64)
    recalls=[]; f1s=[]
    for c in (0,1):
        tp=int(((p==c)&(y==c)).sum()); fn=int(((p!=c)&(y==c)).sum()); fp=int(((p==c)&(y!=c)).sum())
        recalls.append(tp/(tp+fn) if tp+fn else 0.0)
        f1s.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0)
    return {"macro_f1":float(sum(f1s)/2),"balanced_accuracy":float(sum(recalls)/2),
            "auroc":binary_auroc(y,z),"bce":float(bce_sum/len(y))}

def reliability_metrics(labels,auth_logits,rel_logits):
    y=np.asarray(labels,dtype=np.int64); a=np.asarray(auth_logits,dtype=np.float64)
    r=np.asarray(rel_logits,dtype=np.float64); pred=(a>=0).astype(np.int64); correct=(pred==y).astype(np.int64)
    prob=1.0/(1.0+np.exp(-np.clip(r,-80,80)))
    return {"std":float(prob.std()),"brier":float(np.mean((prob-correct)**2)),
            "correctness_auroc":binary_auroc(correct,prob),
            "mean_correct":float(prob[correct==1].mean()) if (correct==1).any() else float("nan"),
            "mean_incorrect":float(prob[correct==0].mean()) if (correct==0).any() else float("nan")}

def checkpoint_key(metrics,epoch):
    return (-float(metrics["macro_f1"]),float(metrics["bce"]),int(epoch))

def _legacy_checkpoint_key(m):
    return (-float(m["macro_f1"]),float(m["auth_bce"]),int(m["epoch"]))

def better_checkpoint(candidate,best):
    """Frozen Step7 dry-contract compatibility API."""
    return best is None or _legacy_checkpoint_key(candidate) < _legacy_checkpoint_key(best)

def planned_run_roots(root=Path("artifacts/post_v1/t1_aegis_vision/training")):
    """Frozen Step7 dry-contract compatibility API."""
    root=Path(root)
    return [root/v/f"seed_{s}" for v in VARIANTS for s in SEEDS]

def protocol_snapshot():
    """Frozen Step7 dry-contract compatibility API."""
    c=TrainingConfig()
    return {
      "variants":list(VARIANTS),"seeds":list(SEEDS),"formal_runs":9,"optimizer":c.optimizer,
      "backbone_lr":c.backbone_lr,"head_lr":c.head_lr,"weight_decay":c.weight_decay,
      "physical_batch":c.physical_batch,"gradient_accumulation":c.grad_accumulation,
      "effective_batch":c.physical_batch*c.grad_accumulation,"gradient_clip_norm":c.grad_clip,
      "reliability_weight":c.reliability_weight,
      "checkpoint_order":["highest validation authenticity Macro-F1","lowest validation authenticity BCE","earliest epoch"],
      "optimizer_steps":0,"training_authorized":False
    }
def validate_authorization(path,runner_commit):
    d=json.loads(Path(path).read_text(encoding="utf-8"))
    required={"schema":"AEGIS_T1_STEP8_TRAINING_AUTHORIZATION_V1","training_authorized":True,
              "runner_commit":runner_commit,"step7d_commit":STEP7D_COMMIT}
    for k,v in required.items():
        if d.get(k)!=v: raise RuntimeError(f"authorization mismatch {k}: {d.get(k)!r} != {v!r}")
    return d

def build_loaders(manifest,raw_root,seed,num_workers,cfg):
    train=UsablePopulationDataset(manifest,raw_root,"train",build_transforms(True))
    val=UsablePopulationDataset(manifest,raw_root,"val",build_transforms(False))
    if len(train)!=323997 or len(val)!=11983: raise RuntimeError(f"population mismatch train={len(train)} val={len(val)}")
    g=torch.Generator(); g.manual_seed(seed)
    common={"batch_size":cfg.physical_batch,"num_workers":num_workers,"pin_memory":True,
            "persistent_workers":num_workers>0,"drop_last":False}
    tr=DataLoader(train,shuffle=True,generator=g,worker_init_fn=seed_worker,**common)
    va=DataLoader(val,shuffle=False,**common)
    return tr,va

@torch.no_grad()
def validate(model,loader,variant,device):
    model.eval(); labels=[]; auth=[]; rel=[]; bce_sum=0.0
    for x,y in loader:
        x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True,dtype=torch.float32)
        with torch.amp.autocast("cuda",dtype=torch.float16):
            out=model(x)
            b=F.binary_cross_entropy_with_logits(out["authenticity_logit"],y,reduction="sum")
        bce_sum+=float(b.item()); labels.extend(y.cpu().tolist()); auth.extend(out["authenticity_logit"].float().cpu().tolist())
        if variant=="A1": rel.extend(out["reliability_logit"].float().cpu().tolist())
    m=authenticity_metrics(labels,auth,bce_sum)
    if variant=="A1":m["reliability"]=reliability_metrics(labels,auth,rel)
    return m

def save_json(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf-8")

def train_one_run(variant,seed,args,runner_commit):
    cfg=TrainingConfig(); seed_everything(seed)
    run_root=Path(args.artifact_root)/variant/f"seed_{seed}"
    if run_root.exists() and any(run_root.iterdir()): raise RuntimeError(f"non-empty run root: {run_root}")
    run_root.mkdir(parents=True,exist_ok=True)
    train_loader,val_loader=build_loaders(args.manifest,args.raw_root,seed,args.num_workers,cfg)
    model=build_model(variant,args.weight).cuda()
    groups=parameter_groups(model,cfg)
    optimizer=torch.optim.AdamW(groups,weight_decay=cfg.weight_decay)
    scaler=torch.amp.GradScaler("cuda")
    updates_per_epoch=math.ceil(len(train_loader)/cfg.grad_accumulation)
    total_updates=updates_per_epoch*cfg.max_epochs
    warmup_updates=updates_per_epoch*cfg.warmup_epochs
    best=None; stale=0; global_update=0; history=[]
    provenance={"variant":variant,"seed":seed,"runner_commit":runner_commit,"step7d_commit":STEP7D_COMMIT,
      "manifest_sha256":sha256(args.manifest),"weight_sha256":sha256(args.weight),
      "train_samples":len(train_loader.dataset),"validation_samples":len(val_loader.dataset),
      "num_workers":args.num_workers,"physical_batch":cfg.physical_batch,"grad_accumulation":cfg.grad_accumulation}
    save_json(run_root/"provenance.json",provenance)
    for epoch in range(1,cfg.max_epochs+1):
        model.train(); optimizer.zero_grad(set_to_none=True); accum_samples=0; epoch_loss_sum=0.0; epoch_samples=0
        for micro,(x,y) in enumerate(train_loader,1):
            x=x.cuda(non_blocking=True); y=y.cuda(non_blocking=True,dtype=torch.float32); bs=int(y.numel())
            with torch.amp.autocast("cuda",dtype=torch.float16):
                out=model(x); losses=compute_losses(out,y,cfg.reliability_weight); mean_loss=losses["total"]
            if not torch.isfinite(mean_loss): raise RuntimeError(f"non-finite loss epoch={epoch} microbatch={micro}")
            scaler.scale(mean_loss*bs).backward(); accum_samples+=bs; epoch_samples+=bs; epoch_loss_sum+=float(mean_loss.detach())*bs
            boundary=(micro%cfg.grad_accumulation==0) or (micro==len(train_loader))
            if boundary:
                scaler.unscale_(optimizer)
                for p in model.parameters():
                    if p.grad is not None:p.grad.div_(accum_samples)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],cfg.grad_clip)
                global_update+=1; apply_lr(optimizer,global_update,warmup_updates,total_updates)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True); accum_samples=0
        metrics=validate(model,val_loader,variant,torch.device("cuda"))
        if not all(math.isfinite(float(metrics[k])) for k in ("macro_f1","balanced_accuracy","auroc","bce")):
            raise RuntimeError(f"non-finite authenticity validation metric epoch={epoch}: {metrics}")
        key=checkpoint_key(metrics,epoch); improved=best is None or key<best["key"]
        if improved:
            best={"key":key,"epoch":epoch,"metrics":metrics}; stale=0
            torch.save({"model":model.state_dict(),"epoch":epoch,"variant":variant,"seed":seed,
                        "metrics":metrics,"runner_commit":runner_commit},run_root/"best.pt")
        else: stale+=1
        row={"epoch":epoch,"train_total_loss":epoch_loss_sum/epoch_samples,"validation":metrics,
             "global_optimizer_updates":global_update,"improved":improved,"stale_epochs":stale}
        history.append(row); save_json(run_root/"history.json",history); save_json(run_root/"best.json",best)
        if epoch>=cfg.min_epochs and stale>=cfg.early_stop_patience:break
    save_json(run_root/"final_summary.json",{"variant":variant,"seed":seed,"best":best,
              "epochs_completed":len(history),"optimizer_steps":global_update,"training_performed":True})
    return run_root

def dry_snapshot():
    c=TrainingConfig()
    return {"variants":list(VARIANTS),"seeds":list(SEEDS),"formal_runs":9,"optimizer":c.optimizer,
      "backbone_lr":c.backbone_lr,"head_lr":c.head_lr,"weight_decay":c.weight_decay,
      "physical_batch":c.physical_batch,"gradient_accumulation":c.grad_accumulation,
      "effective_batch":c.physical_batch*c.grad_accumulation,"gradient_clip_norm":c.grad_clip,
      "reliability_weight":c.reliability_weight,"scheduler":"linear_warmup_then_cosine_per_optimizer_update_to_zero",
      "sample_weighted_accumulation":True,"drop_last":False,
      "checkpoint_order":["highest validation authenticity Macro-F1","lowest validation authenticity BCE","earliest epoch"],
      "optimizer_steps":0,"training_authorized":False}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dry-run",action="store_true")
    ap.add_argument("--execute-training",action="store_true")
    ap.add_argument("--authorization-file")
    ap.add_argument("--variant",choices=VARIANTS)
    ap.add_argument("--seed",type=int,choices=SEEDS)
    ap.add_argument("--num-workers",type=int,default=4)
    ap.add_argument("--manifest",type=Path,default=DEFAULT_MANIFEST)
    ap.add_argument("--raw-root",type=Path,default=DEFAULT_RAW_ROOT)
    ap.add_argument("--weight",type=Path,default=DEFAULT_WEIGHT)
    ap.add_argument("--artifact-root",type=Path,default=DEFAULT_ARTIFACT_ROOT)
    a=ap.parse_args()
    if a.dry_run:
        if a.execute_training: raise SystemExit("dry-run and execute-training are mutually exclusive")
        print(json.dumps(dry_snapshot(),indent=2)); return
    if not a.execute_training: raise SystemExit("REFUSED: training requires --execute-training and a valid Step8 authorization")
    if not a.authorization_file: raise SystemExit("REFUSED: missing --authorization-file")
    if a.variant is None or a.seed is None: raise SystemExit("REFUSED: execute-training requires one explicit --variant and --seed")
    if a.num_workers<0: raise SystemExit("REFUSED: num-workers must be >=0")
    head=git_head(); validate_authorization(a.authorization_file,head)
    if sha256(a.manifest)!=MANIFEST_SHA: raise SystemExit("REFUSED: manifest SHA mismatch")
    if sha256(a.weight)!=WEIGHT_SHA: raise SystemExit("REFUSED: weight SHA mismatch")
    train_one_run(a.variant,a.seed,a,head)

if __name__=="__main__": main()
