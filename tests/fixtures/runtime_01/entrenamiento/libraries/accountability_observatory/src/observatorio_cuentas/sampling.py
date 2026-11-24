"""Reproducible joint-stratified sample; no holdout values inform membership."""
import hashlib
import math
from collections import Counter
import numpy as np
import polars as pl
from agentic_forecasting.data import weighted_entropy


def sample_universe(universe, spec, limit=1000, strict=True, operational_dates=None):
    strata={}
    candidate_count=universe.history["account_id"].n_unique()
    rejected=[]
    axis=sorted(set(operational_dates) if operational_dates is not None else set(universe.history["date"].unique().to_list()))
    if not axis or set(universe.history["date"].unique().to_list()) - set(axis):
        raise ValueError("Source dates absent from operational calendar")
    cutoff=axis[-1]
    pos={d:i for i,d in enumerate(axis)}
    eligible=[]
    for (aid,),g in universe.history.group_by("account_id",maintain_order=True):
        dates=g["date"].to_list()
        indices=[pos[d] for d in dates]
        if g["taxonomy"][0] != "cuentas_cero" and (len(dates)<spec.minimum_history or dates[-1]!=cutoff or indices!=list(range(indices[0],indices[-1]+1))):
            reasons=[]
            if len(dates)<spec.minimum_history:reasons.append("INSUFFICIENT_HISTORY")
            if dates[-1]!=cutoff:reasons.append("STALE_END_DATE")
            if indices!=list(range(indices[0],indices[-1]+1)):reasons.append("INTERNAL_HISTORY_GAPS")
            exclusion={"account_id":aid,"reason":"INELIGIBLE_40_PROTOCOL","details":reasons,"observations":len(dates)}
            universe.exclusions.append(exclusion)
            rejected.append(exclusion)
        else:
            eligible.append(aid)
    universe.history=universe.history.filter(pl.col("account_id").is_in(eligible))
    for (aid,),g in universe.history.group_by("account_id",maintain_order=True):
        prefix=g["balance"].to_numpy()[:max(0,g.height-spec.cv_folds*spec.validation-spec.evaluation_span)]
        budget=min(10,int(weighted_entropy(prefix)*10)+1) if len(prefix)>=3 else 1
        key=(g["taxonomy"][0],budget)
        strata.setdefault(key,[]).append(aid)
    total=sum(map(len,strata.values()))
    if strict and total<limit:
        raise InsufficientEligibleSeries({"requested":limit,"candidates":candidate_count,"eligible":total,"rejected":rejected,"suggested_candidate_limit":max(candidate_count+200, math.ceil(candidate_count*limit/max(total,1)*1.1))})
    target=min(limit,total)
    if not target:
        raise ValueError("Empty source universe")
    keys=sorted(strata)
    # Hamilton allocation proportional to the observed joint distribution.
    quotas={k:math.floor(target*len(strata[k])/total) for k in keys}
    for k in sorted(keys,key=lambda k:(-(target*len(strata[k])/total-quotas[k]),k))[:target-sum(quotas.values())]:
        quotas[k]+=1
    # Preserve represented classes when they exist, while retaining a fixed total.
    for tax in sorted({k[0] for k in keys}):
        if not any(quotas[k] for k in keys if k[0]==tax):
            donors=[k for k in keys if quotas[k]>1]
            if not donors:
                raise ValueError("Sample too small to represent all source classes")
            donor=max(donors,key=lambda k:quotas[k]);recipient=max((k for k in keys if k[0]==tax),key=lambda k:len(strata[k]))
            quotas[donor]-=1;quotas[recipient]+=1
    selected=[];composition=[]
    for k in keys:
        ordered=sorted(strata[k],key=lambda aid:hashlib.sha256(f"{spec.seed}:{aid}".encode()).hexdigest())
        selected.extend(ordered[:quotas[k]])
        composition.append({"taxonomy":k[0],"budget":k[1],"available":len(ordered),"selected":quotas[k]})
    universe.history=universe.history.filter(pl.col("account_id").is_in(selected))
    return {"candidates":candidate_count,"eligibility_rejections":rejected,"requested":limit,"selected":len(selected),"available":total,"seed":spec.seed,"method":"proportional taxonomy/B Hamilton allocation; deterministic hash within strata","composition":composition}


class InsufficientEligibleSeries(ValueError):
    def __init__(self, diagnostic):
        self.diagnostic=diagnostic
        super().__init__(f"Need {diagnostic['requested']} final eligible series; {diagnostic['eligible']} remain from {diagnostic['candidates']} candidates. Inspect sample_diagnostic.json. Increase candidate pool, not the final sample or eligibility thresholds.")
