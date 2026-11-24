"""Post-epoch fixed-set loss measurement. No optimizer or checkpoint selection."""
import numpy as np
import torch
from model_arena.disk import check_space
from model_arena.loss_store import write_parquet
from agentic_forecasting.curve_models import make_batch,CURVE_PROBS

def build_monitor(training,full,end,spec,candidate,device,calendar,fold,cfg,detail_path=None):
    requested=cfg.get('loss_monitor_accounts',128)
    limit=len(training) if requested is None else int(requested)
    if limit<1:raise ValueError('loss_monitor_accounts debe ser positivo')
    rng=np.random.default_rng(spec.seed+901)
    selected=sorted(rng.choice(len(training),min(limit,len(training)),replace=False).tolist())
    origins=rng.integers(max(spec.window,spec.horizon),end-spec.horizon+1,size=len(selected))
    train_refs=list(zip(selected,origins.tolist()))
    validation_refs=[(i,end) for i in selected] if fold!='FUTURO' else []
    epoch_counter=[0]
    if detail_path is not None:
        from pathlib import Path
        detail_path=Path(detail_path)
        detail_path.parent.mkdir(parents=True,exist_ok=True)
        detail_path.mkdir(exist_ok=True)
        detail_path.with_suffix(".parquet").unlink(missing_ok=True)
        # This is called only when the fit has no completed checkpoint.
        for old in detail_path.glob("epoch_*.parquet"):
            old.unlink()
    def measure(model):
        if detail_path is not None:
            check_space(detail_path, stage=f"{candidate.id} {fold} época")
        epoch_counter[0]+=1
        details=[]
        previous=model.training;model.eval()
        def loss(series,refs,split):
            total=np.zeros(2,dtype=np.float64);count=0
            with torch.inference_mode():
                for start in range(0,len(refs),spec.batch_size):
                    features,_,truth=make_batch(series,refs[start:start+spec.batch_size],spec,candidate,device,calendar,True)
                    residual=truth.unsqueeze(-1)-model(features)
                    tau=truth.new_tensor(CURVE_PROBS)
                    elements=torch.maximum(tau*residual,(tau-1)*residual).double()
                    total+=elements.sum(dim=(0,1,3)).cpu().numpy()
                    individual=elements.mean(dim=(1,3)).cpu().numpy()
                    for (index,origin),values in zip(refs[start:start+spec.batch_size],individual):
                        item=series[index]
                        for target,value in zip(['saldo','variacion'],values):
                            details.append(dict(account_id=item.id,currency=item.currency,budget=item.budget,
                                cohort=item.cohort,taxonomy='cuentas_entrenamiento',candidate=candidate.id,
                                fold=fold,target=target,split=split,epoch=epoch_counter[0],loss=float(value),
                                n_observations=spec.horizon*len(CURVE_PROBS)))
                    count+=len(truth)*spec.horizon*len(CURVE_PROBS)
            return total/count if count else [None,None]
        try:
            train=loss(training,train_refs,'train');validation=loss(full,validation_refs,'validation')
            if detail_path is not None and details:
                import polars as pl
                write_parquet(pl.DataFrame(details), detail_path/f"epoch_{epoch_counter[0]:04d}.parquet")
            return dict(fixed_train_saldo=train[0],fixed_train_variacion=train[1],fixed_validation_saldo=validation[0],fixed_validation_variacion=validation[1])
        finally:model.train(previous)
    measure.selection=dict(account_ids=[training[i].id for i in selected],train_origins=[o for i,o in train_refs],validation_origin=end if validation_refs else None,seed=spec.seed+901)
    return measure
