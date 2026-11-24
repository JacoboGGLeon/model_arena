"""Public operations used by the accounting HPO worker."""
import math
from pydantic import BaseModel


def fit(experiment, candidate, series, ends, currency_map, device, *, epochs):
    """Fit a fresh network with fixed epochs and exclusive training endpoints."""
    if len(ends) != len(series) or any(end > len(s.values) for s,end in zip(series,ends)):
        raise ValueError("Invalid series cutoff")
    model, _, _, _, _ = experiment._fit(series,ends,candidate,currency_map,device,tune=False,epochs=epochs)
    return model


def predict(experiment, model, candidate, series, origins, currency_map, device):
    """Return quantiles [series, horizon, target, 3] and separate point predictions."""
    q, _, point = experiment._predict(model,series,origins,candidate,currency_map,device,return_point=True)
    return q, point


class TrialTable(BaseModel):
    trials: list[dict]
    winner: dict | None = None


def tune(trials):
    """Select a configuration from completed expanding-window trials using F1–F3."""
    import agentic_systems as sdk
    def rank(state: TrialTable):
        ranked=[]
        for trial in state.trials:
            folds=trial['folds']
            if sorted(f['fold'] for f in folds)!=[1,2,3] or any(not math.isfinite(f['score']) for f in folds):
                raise ValueError("A trial requires three finite fold scores")
            ranked.append({**trial,'score':sum(f['score'] for f in folds)/3})
        if not ranked:
            raise ValueError("Empty study")
        return {'winner':min(ranked,key=lambda t:(t['score'],t['trial']))}
    graph=sdk.Graph(TrialTable)
    graph.add_node('rank',rank)
    graph.add_edge(sdk.START,'rank')
    graph.add_edge('rank',sdk.END)
    result=sdk.System('forecast_hpo',graph=graph).run({'trials':trials})
    if not result.ok:
        raise ValueError(str(result.errors))
    return result.data['winner']
