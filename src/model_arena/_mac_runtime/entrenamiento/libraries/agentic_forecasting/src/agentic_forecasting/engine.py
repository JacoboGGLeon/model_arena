"""Global forecasting, frozen validation routing and two explicit interval roles."""

import gc
import gzip
import hashlib
import json
import math
import shutil
import time
import uuid
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .contracts import ExperimentSpec
from .data import batch
from .models import BalanceVariationModel, objective
from .surfaces import ForecastSurface

QUANTILES = (0.025, 0.5, 0.975)
INTERVAL_QUANTILES = {0.95: (0, 2)}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(value, allow_nan=False, default=str).encode()
    if path.suffix == ".gz":
        raw = gzip.compress(raw)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(raw)
    tmp.replace(path)


def read_json(path):
    p = Path(path)
    if (
        not p.exists()
        and str(p).endswith(".json.gz")
        and Path(str(p)[:-8] + ".npz").exists()
    ):
        return list(ForecastSurface.load(str(p)[:-8]).rows())
    raw = p.read_bytes()
    return json.loads(gzip.decompress(raw) if p.suffix == ".gz" else raw)


def select_device(requested="auto"):
    if requested not in ("auto", "cpu", "cuda"):
        raise ValueError("Invalid device")
    if requested == "cpu":
        return "cpu"
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        x = torch.tensor([1.0], device="cuda", requires_grad=True)
        (x * x).sum().backward()
        torch.cuda.synchronize()
        return "cuda"
    except Exception:
        if requested == "cuda":
            raise
        return "cpu"


def corrected_quantile(scores, coverage):
    values = np.asarray(scores)
    if len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("Finite out-of-sample calibration scores required")
    rank = math.ceil((len(values) + 1) * coverage)
    if rank > len(values):
        raise ValueError("Insufficient calibration origins for requested coverage")
    return np.sort(values, axis=0)[rank - 1]


def series_fingerprint(series):
    digest = hashlib.sha256()
    for x in series:
        digest.update(
            json.dumps(
                {
                    "id": x.id,
                    "dates": x.dates,
                    "currency": x.currency,
                    "budget": x.budget,
                    "cohort": x.cohort,
                },
                sort_keys=True,
            ).encode()
        )
        digest.update(np.asarray(x.values, dtype="<f8").tobytes())
    return digest.hexdigest()


class Experiment:
    def __init__(self, spec=None, output="runs/arena", calendar_table=None):
        self.spec = spec or ExperimentSpec()
        self.output = Path(output)
        self.calendar_table = calendar_table

    def model(self, candidate, currency_vocab=1, device="cpu"):
        m = BalanceVariationModel(
            "A" if candidate.encoder_design == "separate" else "B",
            {"tft_compact": "tft", "tft_full": "tft"}.get(candidate.architecture, candidate.architecture),
            self.spec.window,
            self.spec.horizon,
            0.1,
            input_bundle=candidate.inputs,
            currency_vocab=currency_vocab,
            calendar_features=self.spec.calendar_features,
            coherence_weight=candidate.coherence_weight,
        ).to(device)
        if candidate.architecture == "tft_full":
            from .full_tft import FullTFTEncoder
            m.encoders=nn.ModuleList(FullTFTEncoder(self.spec.window,self.spec.horizon,candidate.inputs,self.spec.calendar_features) for _ in m.encoders).to(device)
        if candidate.recipe == "mac_modeling":
            for head in m.heads:
                head[-1] = nn.Linear(96, 3).to(device)
            m.quantile_mode = True
        m.decoder=candidate.decoder
        if candidate.decoder=="residual":
            m.point_heads=nn.ModuleList(nn.Linear(96,1) for _ in range(2)).to(device)
            m.global_heads=nn.ModuleList(nn.Linear(96,1) for _ in range(2)).to(device)
            m.residual_heads=nn.ModuleList(nn.Sequential(nn.Linear(97,96),nn.GELU(),nn.Dropout(.1),nn.Linear(96,3)) for _ in range(2)).to(device)
            # Remove unused direct heads from residual candidates.
            m.heads=nn.ModuleList()
        return m

    def _point(self, model, b):
        value = model(b)
        return value[..., 1] if value.ndim == 4 else value

    def _predict(self, model, series, refs, candidate, currency_map, device, mc=False, return_point=False):
        model.eval()
        if mc:
            for module in model.modules():
                if isinstance(module, nn.Dropout):
                    module.train()
        points = []
        outputs = []
        scales = []
        with torch.inference_mode():
            for start in range(0, len(refs), self.spec.batch_size):
                b = batch(
                    series,
                    refs[start : start + self.spec.batch_size],
                    self.spec,
                    candidate,
                    device,
                    currency_map,
                    observed=False,
                    calendar_table=self.calendar_table,
                )
                value = model(b)
                scale = b["scales"][:, None, :]
                if return_point:
                    point=model.point_scaled * scale
                    if not torch.isfinite(point).all():raise ValueError("Nonfinite point forecast")
                    points.append(point.cpu().numpy())
                if value.ndim == 4:
                    scale = scale.unsqueeze(-1)
                value = torch.sinh(value) * scale
                if not torch.isfinite(value).all():
                    raise ValueError("Nonfinite forecast")
                outputs.append(value.cpu().numpy())
                scales.append(b["scales"].cpu().numpy())
        model.eval()
        result=(np.concatenate(outputs),np.concatenate(scales))
        return (*result,np.concatenate(points)) if return_point else result

    def _fit(
        self, series, ends, candidate, currency_map, device, tune=True, epochs=None
    ):
        s = self.spec
        torch.manual_seed(s.seed)
        rng = np.random.default_rng(s.seed)
        model = self.model(candidate, max(1, len(currency_map)), device)
        refs = [
            (i, o)
            for i, x in enumerate(series)
            if x.cohort == "seen"
            for o in range(s.window, ends[i] - s.horizon + 1)
        ]
        if not refs:
            raise ValueError("No gradient training windows available")
        gradient_ids = sorted({series[i].id for i, _ in refs})
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=s.learning_rate, weight_decay=0.0001
        )
        groups = {}
        for ref in refs:
            groups.setdefault(series[ref[0]].budget, []).append(ref)
        budgets = sorted(groups)
        by_account = {}
        for ref in refs:
            by_account.setdefault(ref[0], []).append(ref)
        # A full-universe run must visit every seen account at least once.
        effective_samples = max(s.samples_per_epoch, 5*math.ceil(len(by_account)/s.batch_size)*s.batch_size)
        print(f"Training samples/epoch: {effective_samples}; seen accounts: {len(by_account)}", flush=True)
        log = []
        best = float("inf")
        best_state = None
        best_epoch = 1
        best_sampled_ids = set()
        stale = 0
        sampled_ids = set()
        validation = (
            [
                (i, ends[i])
                for i, x in enumerate(series)
                if x.id in gradient_ids and ends[i] + s.horizon <= len(x.values)
            ]
            if tune
            else []
        )
        # A fixed training-prefix probe separates genuine progress from changing batches.
        probe_refs=[by_account[i][len(by_account[i])//2] for i in sorted(by_account)]
        stride=max(1,math.ceil(len(probe_refs)/s.batch_size))
        probe_refs=probe_refs[::stride]
        for epoch in range(1, (epochs or s.epochs) + 1):
            model.train()
            point_losses = []
            losses = []
            sample = [
                group[int(rng.integers(len(group)))] for group in by_account.values()
            ]
            for j in range(effective_samples - len(sample)):
                group = groups[budgets[j % len(budgets)]]
                sample.append(group[int(rng.integers(len(group)))])
            rng.shuffle(sample)
            if candidate.approach=="curriculum":
                # A persistent model and optimizer; budgets enter cumulatively, with final balanced replay.
                selected=[]
                batch_order=[]
                for start in range(0,effective_samples,s.batch_size):
                    progress=((epoch-1)*effective_samples+start)/max(1,(epochs or s.epochs)*effective_samples)
                    gate=budgets[min(len(budgets)-1,int(progress/.8*len(budgets)))]
                    pool=[aid for aid in sorted(by_account) if series[aid].budget<=gate]
                    if progress>=.8:
                        if not hasattr(model,"_replay_cursor"):model._replay_cursor=0
                        account_ids=[sorted(by_account)[(model._replay_cursor+j)%len(by_account)] for j in range(min(s.batch_size,effective_samples-start))]
                        model._replay_cursor+=len(account_ids)
                    else:
                        account_ids=[pool[int(rng.integers(len(pool)))] for _ in range(min(s.batch_size,effective_samples-start))]
                    chunk=[by_account[i][int(rng.integers(len(by_account[i])))] for i in account_ids]
                    selected.extend(chunk);batch_order.append(dict(step=start//s.batch_size,max_budget=gate,replay=progress>=.8,counts={str(b):sum(series[i].budget==b for i,o in chunk) for b in budgets}))
                sample=selected
            else:batch_order=[]
            epoch_ids = {series[i].id for i, _ in sample}
            sampled_ids.update(epoch_ids)
            for start in range(0, len(sample), s.batch_size):
                b = batch(
                    series,
                    sample[start : start + s.batch_size],
                    s,
                    candidate,
                    device,
                    currency_map,
                    calendar_table=self.calendar_table,
                )
                optimizer.zero_grad()
                value = model(b)
                loss, _ = objective(
                    value, b, model.experiment, candidate.coherence_weight, getattr(model,"last_global",None)
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if hasattr(model,"point_heads"):
                    # Linear output is in original amount / prefix scale. Detached
                    # features keep the point objective out of the quantile gradient.
                    point_loss=torch.nn.functional.mse_loss(model.point_scaled,b["future_y"].sinh())
                    if not torch.isfinite(point_loss):raise ValueError("Nonfinite scaled point MSE")
                    point_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.point_heads.parameters(),1.0)
                    point_losses.append(point_loss.item())
                optimizer.step()
                losses.append(float(loss.detach()))
            probe_loss=None
            if tune:
                model.eval()
                with torch.no_grad():
                    pb=batch(series,probe_refs,s,candidate,device,currency_map,calendar_table=self.calendar_table)
                    pv=model(pb)
                    probe_loss=float(objective(pv,pb,model.experiment,candidate.coherence_weight,getattr(model,"last_global",None))[0])
            score = float(np.mean(losses))
            if validation:
                predictions, _ = self._predict(
                    model, series, validation, candidate, currency_map, device
                )
                if predictions.ndim == 4:
                    predictions = predictions[..., 1]
                errors = {}
                for k, (i, o) in enumerate(validation):
                    denominator = np.maximum(
                        np.mean(np.abs(np.diff(series[i].values[:o], axis=0)), axis=0),
                        1.0,
                    )
                    error = float(
                        np.mean(
                            np.abs(predictions[k] - series[i].values[o : o + s.horizon])
                            / denominator
                        )
                    )
                    errors.setdefault(series[i].budget, []).append(error)
                score = float(np.mean([np.mean(v) for v in errors.values()]))
            log.append(
                {
                    "epoch": epoch,
                    "point_scaled_mse": float(np.mean(point_losses)) if point_losses else None,
                    "samples_per_epoch": effective_samples,
                    "train_loss": float(np.mean(losses)),
                    "fixed_loss": probe_loss,
                    "fixed_windows":len(probe_refs),
                    "optimizer_steps": math.ceil(effective_samples/s.batch_size),
                    "curriculum": batch_order,
                    "validation_score": score,
                    "sampled_account_ids": sorted(epoch_ids),
                }
            )
            if score < best and (candidate.approach!="curriculum" or epoch>=min((epochs or s.epochs),math.floor(.8*(epochs or s.epochs))+1)):
                best = score
                best_epoch = epoch
                best_sampled_ids = set(sampled_ids)
                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
                stale = 0
            else:
                stale += 1
            # All approaches receive the same predeclared update budget.
        if tune:
            model.load_state_dict(best_state)
        model.sampled_account_ids = sorted(best_sampled_ids if tune else sampled_ids)
        return model, gradient_ids, log, best_epoch, best

    def run(self, series, device="auto", resume=False):
        s = self.spec
        device = select_device(device)
        torch.set_num_threads(s.cpu_threads)
        if not series or len({x.id for x in series}) != len(series):
            raise ValueError("Nonempty unique series required")
        for x in series:
            v = np.asarray(x.values)
            if v.ndim != 2 or v.shape[1] != 2 or not np.isfinite(v).all():
                raise ValueError("Finite [N,2] values required")
            if len(v) < s.minimum_history or len(x.dates) != len(v)+s.horizon or x.dates != sorted(set(x.dates)):
                raise ValueError("Insufficient history or invalid dates")
            if x.cohort not in ("seen", "zero_shot") or not 1 <= x.budget <= 10:
                raise ValueError("Invalid cohort/budget")
        if not any(x.cohort == "seen" for x in series):
            raise ValueError("No seen accounts")
        if len({tuple(x.dates[len(x.values)-s.test:len(x.values)]) for x in series}) != 1:
            raise ValueError("Common evaluation dates required")
        existing=self.output.exists() and any(self.output.iterdir())
        if existing:
            if not resume:raise FileExistsError("Arena no vacía: usa resume explícito")
            previous=read_json(self.output/"experiment.json")
            if previous["spec"]!=json.loads(json.dumps(s.to_dict())) or previous["input_sha256"]!=series_fingerprint(series):
                raise ValueError("Resume rechazado: spec o insumos cambiaron")
        self.output.mkdir(parents=True, exist_ok=True)
        currency_map = {c:i+1 for i,c in enumerate(sorted({x.currency for x in series if x.cohort=="seen"}))}
        write_json(self.output/"experiment.json", {"spec":s.to_dict(),"input_sha256":series_fingerprint(series),"protocol":"quantile-consecutive-frozen-v2","device":device})
        receipts=[]
        for number,c in enumerate(s.candidates(),1):
            destination=self.output/c.id
            if existing and destination.exists():
                receipt_path=destination/"receipt.json"
                if receipt_path.exists():
                    saved=read_json(receipt_path)
                    if saved["candidate_id"]!=c.id:raise ValueError("Receipt inconsistente")
                    for phase in ("test","horizon"):
                        loaded=ForecastSurface.load(destination/phase)
                        if loaded.metadata["candidate_id"]!=c.id:raise ValueError("Superficie inconsistente")
                    receipts.append(saved);print(f"[{number}/{len(s.candidates())}] REUSED {c.id}",flush=True);continue
                archive=self.output.parent/"resume_incomplete";archive.mkdir(exist_ok=True)
                destination.rename(archive/(c.id+"_"+uuid.uuid4().hex[:8]))
            folds=[]
            for fold in range(s.cv_folds,0,-1):
                tune_ends=[len(x.values)-s.evaluation_span-fold*s.validation for x in series]
                tuning,gradient_ids,history,best_epoch,score=self._fit(series,tune_ends,c,currency_map,device)
                refs=[(i,e) for i,e in enumerate(tune_ends)]
                val,_=self._predict(tuning,series,refs,c,currency_map,device)
                evidence_dir=self.output/'fold_surfaces'/c.id
                evidence_dir.mkdir(parents=True,exist_ok=True)
                fold_number=s.cv_folds-fold+1
                np.savez_compressed(evidence_dir/f'fold_{fold_number}.npz',quantiles=val,observed=np.stack([series[i].values[e:e+s.horizon] for i,e in refs]))
                write_json(evidence_dir/f'fold_{fold_number}.json',dict(series_ids=[x.id for x in series],origins=[x.dates[e-1] for x,e in zip(series,tune_ends)],dates=[x.dates[e:e+s.horizon] for x,e in zip(series,tune_ends)],training_ends=tune_ends))

                account_scores={}
                for k,(i,end) in enumerate(refs):
                    if series[i].id in gradient_ids:
                        denom=np.maximum(np.mean(np.abs(np.diff(series[i].values[:end],axis=0)),axis=0),1.0)
                        account_scores[series[i].id]=float(np.mean(np.abs(val[k,...,1]-series[i].values[end:end+s.horizon])/denom))
                from .percentage_metrics import percentage_metrics
                cv_metrics=[]
                for currency in sorted(currency_map):
                    for budget in sorted({x.budget for x in series}):
                        indices=[i for i,x in enumerate(series) if x.id in gradient_ids and x.currency==currency and x.budget==budget]
                        if not indices: continue
                        for t,target in enumerate(('saldo','variacion')):
                            y=np.stack([series[i].values[tune_ends[i]:tune_ends[i]+s.horizon,t] for i in indices]).astype(float)
                            q=val[indices,:,t,:]; point=q[...,1]; error=y-point
                            cv_metrics.append(dict(candidate_id=c.id,phase='validacion',target=target,currency=currency,budget=budget,backtest_cut=s.cv_folds-fold+1,count=y.size,mae=float(np.abs(error).mean()),rmse=float(np.sqrt((error**2).mean())),observed_coverage=float(((q[...,0]<=y)&(y<=q[...,2])).mean()),mean_width=float((q[...,2]-q[...,0]).mean()),**percentage_metrics(y,point)))
                del tuning
                folds.append(dict(score=score,epoch=best_epoch,metrics=cv_metrics,accounts=account_scores,history=history))
            score=float(np.mean([f['score'] for f in folds]))
            best_epoch=max(1,round(np.mean([f['epoch'] for f in folds])))
            account_scores={aid:float(np.mean([f['accounts'][aid] for f in folds])) for aid in folds[0]['accounts']}
            history=folds
            ends=[len(x.values)-s.evaluation_span for x in series]
            model,gradient_ids,_,_,_=self._fit(series,ends,c,currency_map,device,tune=False,epochs=s.epochs)
            path=self.output/c.id;path.mkdir()
            torch.save({"state_dict":model.state_dict(),"candidate":asdict(c),"spec":s.to_dict(),"currency_map":currency_map,"training_ends":dict(zip([x.id for x in series],ends)),"gradient_account_ids":model.sampled_account_ids},path/"evaluation.pt")
            for cut in range(s.backtest_cuts):
                for phase in ("test","horizon"):
                    surface=self._surface40(model,series,c,currency_map,device,phase,cut)
                    destination=path/'backtest'/f'cut_{cut+1:03d}'
                    destination.mkdir(parents=True,exist_ok=True)
                    surface.save(destination/phase)
                    if cut==s.backtest_cuts-1:
                        surface.save(path/phase)
            receipt={"candidate":asdict(c),"candidate_id":c.id,"selection_score":score,"account_selection_scores":account_scores,"best_epoch":best_epoch,"training_history":history,"gradient_account_ids":model.sampled_account_ids,"gradient_pool_account_ids":gradient_ids,"zero_shot_ids":[x.id for x in series if x.cohort=="zero_shot"]}
            write_json(path/"receipt.json",receipt);receipts.append(receipt)
            del model;gc.collect()
            print(f"[{number}/{len(s.candidates())}] {c.id}",flush=True)
        winners={"mac_modeling":{x.id:min(receipts,key=lambda r:(r["account_selection_scores"].get(x.id,r["selection_score"]),r["candidate_id"]))["candidate_id"] for x in series}}
        budgets=sorted({x.budget for x in series if x.cohort=="seen"})
        budget_scores={str(b):{r["candidate_id"]:float(np.mean([r["account_selection_scores"][x.id] for x in series if x.cohort=="seen" and x.budget==b])) for r in receipts} for b in budgets}
        budget_winners={b:min(scores,key=lambda c:(scores[c],c)) for b,scores in budget_scores.items()}
        write_json(self.output/"budget_winners.json",dict(winners=budget_winners,scores=budget_scores,criterion="Mean per-account fold-normalized median error; validation seen only"))
        write_json(self.output/"winners.json",winners)
        write_json(self.output/"result.json",{"status":"COMPLETED","protocol":"quantile-consecutive-frozen-v2","candidates":len(receipts),"winners":winners,"routing":"Validation only; zero-shot uses pooled seen validation"})
        return ExperimentResult(self.output)

    def _surface40(self,model,series,c,currency_map,device,phase,cut=0):
        s=self.spec
        if phase not in ("test","horizon"):
            raise ValueError("Unknown evaluation phase")
        starts=[len(x.values)-s.evaluation_span+cut*s.backtest_stride+(s.test if phase=="horizon" else 0) for x in series]
        refs=list(enumerate(starts))
        values,_=self._predict(model,series,refs,c,currency_map,device)
        origins=[[x.dates[o-1]]*s.horizon for x,o in zip(series,starts)]
        point=values[...,1]
        low=np.stack([values[...,INTERVAL_QUANTILES[cov][0]] for cov in s.coverages])[None]
        high=np.stack([values[...,INTERVAL_QUANTILES[cov][1]] for cov in s.coverages])[None]
        observed=np.stack([x.values[o:o+s.horizon] for x,o in zip(series,starts)])
        meta={"format":"forecast-npz-v2","series_ids":[x.id for x in series],"cohorts":[x.cohort for x in series],"budgets":[x.budget for x in series],"currencies":[x.currency for x in series],"coverages":list(s.coverages),"candidate_id":c.id,"recipe":c.recipe,"phase":phase,"origins":[v[0] for v in origins],"origins_by_step":origins,"target_dates":[x.dates[o:o+s.horizon] for x,o in zip(series,starts)],"backtest_cut":cut+1,"roles":["operational"],"origin_mode":"fixed"}
        return ForecastSurface(meta,point,observed,low,high)


class ExperimentResult:
    def __init__(self,path):
        self.path=Path(path)

    def compare(self):
        output=[]
        for folder in sorted(self.path.iterdir()):
            if not folder.is_dir() or not (folder/"receipt.json").exists():
                continue
            for surface_path in sorted((folder/'backtest').glob('cut_*/*.meta.json')):
                phase=surface_path.name.split('.')[0]
                f=ForecastSurface.load(surface_path.with_name(phase));m=f.metadata
                groups={}
                for i,key in enumerate(zip(m["budgets"],m["cohorts"],m["currencies"])):
                    groups.setdefault(key,[]).append(i)
                for (budget,cohort,currency),indices in groups.items():
                    for t,target in enumerate(("saldo","variacion")):
                        for h in range(f.point.shape[1]):
                            y=f.observed[indices,h,t];p=f.point[indices,h,t]
                            lo=f.lower[0,0,indices,h,t];hi=f.upper[0,0,indices,h,t]
                            output.append({"candidate_id":folder.name,"recipe":m["recipe"],"phase":phase,"cohort":cohort,"budget":budget,"currency":currency,"target":target,"position":h+1,"backtest_cut":m["backtest_cut"],"horizon":h+1,"role":"operational","coverage":.95,"count":len(indices),"mae":float(np.mean(np.abs(y-p))),"rmse":float(np.sqrt(np.mean((y-p)**2))),"observed_coverage":float(np.mean((lo<=y)&(y<=hi))),"mean_width":float(np.mean(hi-lo)),"interval_score":float(np.mean(hi-lo+40*np.maximum(lo-y,0)+40*np.maximum(y-hi,0)))})
                            from .percentage_metrics import percentage_metrics
                            output[-1].update(percentage_metrics(y,p))
        return output
