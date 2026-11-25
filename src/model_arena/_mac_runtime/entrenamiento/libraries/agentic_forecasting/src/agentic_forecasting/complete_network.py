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

from model_arena.contracts import Contract

FAMILIES = ('tide', 'nhits', 'nlinear', 'dlinear', 'nbeats')
UPSTREAM = '3.1.2'


class CompleteParameters(Contract):
    lr: float = Field(default=.001, gt=0, allow_inf_nan=False)
    weight_decay: float = Field(default=1e-5, ge=0, allow_inf_nan=False)
    architecture: dict = Field(default_factory=dict)


def capabilities(kind, historical, future):
    if kind not in FAMILIES:
        raise ValueError('Unknown complete forecasting family: '+kind)
    if kind in ('nlinear', 'dlinear', 'nbeats') and (historical or future):
        raise ValueError(kind+' supports X0 only; calendar inputs are not silently discarded')


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
    elif kind != 'nlinear':
        raise ValueError('No search adapter for '+kind)
    return dict(architecture=architecture,lr=trial.suggest_float('lr',*space.lr_range,log=True),
                weight_decay=trial.suggest_float('weight_decay',*space.wd_range,log=True))


class CompleteNetwork(nn.Module):
    def __init__(self, *, family, targets, window, h, historical, future, parameters):
        super().__init__()
        capabilities(family,historical,future)
        if version('neuralforecast') != UPSTREAM:
            raise RuntimeError('This adapter requires neuralforecast=='+UPSTREAM)
        from neuralforecast.models import TiDE, NHITS, NLinear, DLinear, NBEATS
        from neuralforecast.losses.pytorch import MQLoss
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
                loss=MQLoss(quantiles=[.025,.5,.975]),scaler_type='identity',
                random_seed=(torch.initial_seed()+i) % (2**32),
                **parameters.architecture) for i in range(targets))
        finally:
            np.random.set_state(numpy_state)
            random.setstate(python_state)
        self.provenance=dict(provider='neuralforecast',version=UPSTREAM,family=family,
            source_sha256=hashlib.sha256(Path(inspect.getfile(cls)).read_bytes()).hexdigest(),
            adapter_sha256=hashlib.sha256(inspect.getsource(CompleteNetwork).encode()).hexdigest(),
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
            raw=model(dict(insample_y=y,insample_mask=torch.ones_like(y),hist_exog=past,
                           futr_exog=futr,stat_exog=None))
            predictions.append(raw.sort(dim=-1).values)
        return torch.stack(predictions,dim=2)
