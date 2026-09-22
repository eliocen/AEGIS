from dataclasses import dataclass
@dataclass(frozen=True)
class TrainingConfig:
    optimizer:str="AdamW"; backbone_lr:float=1e-5; head_lr:float=1e-4; weight_decay:float=0.05
    warmup_epochs:int=1; max_epochs:int=20; min_epochs:int=5; early_stop_patience:int=4
    physical_batch:int=16; grad_accumulation:int=4; grad_clip:float=1.0; amp:str="fp16"
    seeds:tuple=(42,43,44); reliability_weight:float=0.25
