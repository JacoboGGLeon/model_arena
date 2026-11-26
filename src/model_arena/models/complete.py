"""Pinned upstream architectures with Model Arena's causal training contract."""
import hashlib
import inspect
import random
import numpy as np
from pathlib import Path
from importlib.metadata import version

import torch
from torch import nn
from pydantic import Field

from ..contracts import Contract
from .tft import TFTCompact

FAMILIES = ('tide', 'nhits', 'nlinear', 'dlinear', 'nbeats', 'rlinear')
UPSTREAM = '3.1.2'
# Identity emitted by the unchanged legacy percentile adapter in 0.1.7.
LEGACY_ADAPTER_SHA = 'a430878d6ef91e23edaab2948e2eeec17eafef851bfdc3ca76c78d4b8db09aa9'


class CompleteParameters(Contract):
    lr: float = Field(default=.001, gt=0, allow_inf_nan=False)
    weight_decay: float = Field(default=1e-5, ge=0, allow_inf_nan=False)
    architecture: dict = Field(default_factory=dict)


def capabilities(kind, historical, future):
    if kind not in FAMILIES:
        raise ValueError('Unknown complete forecasting family: '+kind)
    if kind in ('nlinear', 'dlinear', 'nbeats', 'rlinear') and (historical or future):
        raise ValueError(kind+' supports history-only inputs (such as X0); historical and future calendar features are unsupported')


def suggest_parameters(kind, trial, space):
    architecture = {}
    if kind == 'tide':
        architecture = dict(hidden_size=trial.suggest_categorical('hidden_size', space.width),
            num_encoder_layers=trial.suggest_categorical('num_encoder_layers', space.depth),
            num_decoder_layers=trial.suggest_categorical('num_decoder_layers', space.decoder_depth),
            decoder_output_dim=trial.suggest_categorical('decoder_output_dim', [8,16,32]),
            temporal_decoder_dim=trial.suggest_categorical('temporal_decoder_dim', space.width),
            dropout=space.dropout)
    elif kind == 'nhits':
        width=trial.suggest_categorical('hidden_size',space.width)
        blocks=trial.suggest_categorical('blocks_per_stack',[1,2])
        architecture=dict(n_blocks=[blocks]*3,mlp_units=[[width,width]]*3,
            n_pool_kernel_size=trial.suggest_categorical('pooling',[1,2,4]),
            activation=trial.suggest_categorical('activation',['ReLU','Tanh']),
            dropout_prob_theta=space.dropout)
        k=architecture.pop('n_pool_kernel_size')
        architecture['n_pool_kernel_size']=[k,k,1]
    elif kind == 'nbeats':
        width=trial.suggest_categorical('hidden_size',space.width)
        blocks=trial.suggest_categorical('blocks_per_stack',[1,2])
        architecture=dict(n_blocks=[blocks]*3,mlp_units=[[width,width]]*3,
            shared_weights=trial.suggest_categorical('shared_weights',[False,True]),
            activation=trial.suggest_categorical('activation',['ReLU','Tanh']))
    elif kind == 'dlinear':
        architecture=dict(moving_avg_window=trial.suggest_categorical('moving_avg_window',[3,5,9,25]))
    elif kind == 'rlinear':
        architecture = dict(dropout=trial.suggest_categorical('dropout', [0., space.dropout]) if space.dropout else 0.)
    elif kind != 'nlinear':
        raise ValueError('No search adapter for '+kind)
    return dict(architecture=architecture,lr=trial.suggest_float('lr',*space.lr_range,log=True),
                weight_decay=trial.suggest_float('weight_decay',*space.wd_range,log=True))


class RLinear(nn.Module):
    """Channel-independent linear projection with reversible input normalization."""

    def __init__(self, window, horizon, outputs, dropout=0.0):
        super().__init__()
        self.horizon, self.outputs = horizon, outputs
        self.projection = nn.Linear(window, horizon * outputs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values):
        center = values.mean(dim=1, keepdim=True)
        spread = (values.var(dim=1, keepdim=True, unbiased=False) + 1e-5).sqrt()
        normalized = (values - center) / spread
        forecast = self.projection(self.dropout(normalized.squeeze(-1)))
        return forecast.reshape(-1, self.horizon, self.outputs) * spread + center


class CompleteNetwork(nn.Module):
    def __init__(self, *, family, targets, window, h, historical, future, parameters, point=False):
        super().__init__()
        capabilities(family,historical,future)
        self.window,self.h,self.historical,self.future=window,h,historical,future
        if family == 'rlinear':
            if set(parameters.architecture) - {'dropout'}:
                raise ValueError('Unsupported RLinear architecture parameters')
            self.models = nn.ModuleList(
                RLinear(window, h, 1 if point else 3, **parameters.architecture)
                for _ in range(targets)
            )
            self.provenance = dict(provider='model_arena', family=family,
                source_sha256=hashlib.sha256(inspect.getsource(RLinear).encode()).hexdigest(),
                objective='point' if point else 'quantiles',
                reference='https://arxiv.org/abs/2305.10721')
            return
        if version('neuralforecast') != UPSTREAM:
            raise RuntimeError('This adapter requires neuralforecast=='+UPSTREAM)
        from neuralforecast.models import TiDE, NHITS, NLinear, DLinear, NBEATS
        from neuralforecast.losses.pytorch import MAE, MQLoss
        cls=dict(tide=TiDE,nhits=NHITS,nlinear=NLinear,dlinear=DLinear,nbeats=NBEATS)[family]
        allowed=dict(tide={'hidden_size','num_encoder_layers','num_decoder_layers','decoder_output_dim',
                          'temporal_decoder_dim','dropout','temporal_width','layernorm'},
                     nhits={'n_blocks','mlp_units','n_pool_kernel_size','n_freq_downsample',
                            'pooling_mode','interpolation_mode','activation','dropout_prob_theta'},
                     nlinear=set(),dlinear={'moving_avg_window'},
                     nbeats={'n_harmonics','n_basis','basis','stack_types','n_blocks','mlp_units',
                             'dropout_prob_theta','activation','shared_weights'})[family]
        if set(parameters.architecture)-allowed:
            raise ValueError('Unsupported '+family+' architecture parameters')
        self.window,self.h,self.historical,self.future=window,h,historical,future
        exogenous=dict(hist_exog_list=[f'h{i}' for i in range(historical)] or None,
                       futr_exog_list=[f'f{i}' for i in range(future)] or None) if family in ('tide','nhits') else {}
        numpy_state,python_state=np.random.get_state(),random.getstate()
        try:
            self.models=nn.ModuleList(cls(h=h,input_size=window,
                **exogenous,
                loss=MAE() if point else MQLoss(quantiles=[.025,.5,.975]),scaler_type='identity',
                random_seed=(torch.initial_seed()+i) % (2**32),
                **parameters.architecture) for i in range(targets))
        finally:
            np.random.set_state(numpy_state)
            random.setstate(python_state)
        self.provenance=dict(provider='neuralforecast',version=UPSTREAM,family=family,
            source_sha256=hashlib.sha256(Path(inspect.getfile(cls)).read_bytes()).hexdigest(),
            adapter_sha256=LEGACY_ADAPTER_SHA if not point else hashlib.sha256(inspect.getsource(CompleteNetwork).encode()).hexdigest(),
            targets='independent_networks',history_transform='linear_window_normalized')

    def forward(self, history, past, future):
        if self.historical and (past is None or past.shape[1:] != (self.window,self.historical)):
            raise ValueError('Historical covariate shape mismatch')
        if self.future and (future is None or future.shape[1:] != (self.h,self.future)):
            raise ValueError('Future covariate shape mismatch')
        # The existing trainer passes asinh(history / scale); upstream consumes linear history / scale.
        linear=history.sinh()
        futr=torch.cat((future.new_zeros(len(history),self.window,self.future),future),dim=1) if self.future else None
        predictions=[]
        for i,model in enumerate(self.models):
            y=linear[:,:,i:i+1]
            raw = model(y) if isinstance(model, RLinear) else model(dict(
                insample_y=y, insample_mask=torch.ones_like(y), hist_exog=past,
                futr_exog=futr, stat_exog=None))
            if raw.ndim == 2:
                raw = raw.unsqueeze(-1)
            predictions.append(raw.sort(dim=-1).values)
        return torch.stack(predictions,dim=2)


class CompleteModel(TFTCompact):
    def __init__(self, model_spec, arena_spec):
        self.model_spec,self.spec=model_spec,arena_spec
        self.params=CompleteParameters.model_validate(model_spec.params)
        self._feature_cache={}
        self.device=torch.device(arena_spec.training.device)
        torch.set_num_threads(arena_spec.training.cpu_threads)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(arena_spec.training.seed)
            self.network=CompleteNetwork(family=model_spec.kind,targets=len(arena_spec.schema.target_cols),
                window=arena_spec.window,h=arena_spec.h,historical=len(model_spec.historical_features),
                future=len(model_spec.future_features),parameters=self.params).to(self.device)

    def save(self,path):
        torch.save(dict(state_dict=self.network.state_dict(),provenance=self.network.provenance),path)

    def restore(self,path):
        saved=torch.load(path,map_location=self.device,weights_only=True)
        if saved['provenance'] != self.network.provenance:
            raise ValueError('Upstream architecture provenance changed')
        self.network.load_state_dict(saved['state_dict'])
        return self
