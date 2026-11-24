"""TFT completo adaptado al contrato pareado: sin covariables estáticas adicionales.
Reference: Lim et al. arXiv:1912.09363. Shared-value interpretable causal attention.
Not an official/reference-library implementation. Same 96 hidden units as compact.
"""
import math
import torch
from torch import nn

class GRN(nn.Module):
    def __init__(self, n, context=False):
        super().__init__(); self.a=nn.Linear(n,n); self.b=nn.Linear(n,n)
        self.c=nn.Linear(n,n,bias=False) if context else None
        self.gate=nn.Sequential(nn.Dropout(.1),nn.Linear(n,2*n),nn.GLU())
        self.norm=nn.LayerNorm(n)
    def forward(self,x,c=None):
        z=self.a(x)
        if self.c is not None and c is not None:
            while c.ndim<z.ndim:c=c.unsqueeze(1)
            z=z+self.c(c)
        return self.norm(x+self.gate(self.b(torch.nn.functional.elu(z))))

class Selection(nn.Module):
    def __init__(self,variables,n):
        super().__init__();self.variables=variables
        self.emb=nn.ModuleList(nn.Linear(1,n) for _ in range(variables))
        self.transforms=nn.ModuleList(GRN(n) for _ in range(variables))
        self.weight=nn.Sequential(nn.Linear(variables*n,n),nn.ELU(),nn.Linear(n,variables))
        self.context=nn.Linear(n,variables,bias=False)
    def forward(self,x,c):
        emb=[e(x[...,i:i+1]) for i,e in enumerate(self.emb)]
        w=(self.weight(torch.cat(emb,-1))+self.context(c)[:,None,:]).softmax(-1)
        self.last_weights=w.detach()
        return (torch.stack([g(e) for g,e in zip(self.transforms,emb)],-2)*w.unsqueeze(-1)).sum(-2)

class FullTFTEncoder(nn.Module):
    def __init__(self,window,horizon,bundle,features,n=96,heads=4):
        super().__init__(); self.bundle=bundle;self.horizon=horizon;self.window=window
        self.position=nn.Embedding(horizon,n)
        # No account ID, currency or budget embedding added to either competitor.
        # Neutral shared context when the input contract has no static covariates.
        self.neutral=nn.Parameter(torch.zeros(n))
        self.contexts=nn.ModuleList(GRN(n) for _ in range(4))
        self.history_selection=Selection(2+(features if bundle in ('X1','X3') else 0),n)
        self.future_selection=Selection(features,n) if bundle in ('X2','X3') else None
        self.past_calendar=nn.Sequential(nn.Linear(window*features,n),nn.GELU(),nn.Linear(n,n)) if bundle in ('X1','X3') else None
        self.encoder=nn.LSTM(n,n,batch_first=True); self.decoder=nn.LSTM(n,n,batch_first=True)
        self.local_gate=nn.Sequential(nn.Dropout(.1),nn.Linear(n,n*2),nn.GLU()); self.local_norm=nn.LayerNorm(n)
        self.enrich=GRN(n,True);self.q=nn.Linear(n,n);self.k=nn.Linear(n,n)
        self.v=nn.Linear(n,n//heads);self.out=nn.Linear(n//heads,n);self.heads=heads
        self.att_gate=nn.Sequential(nn.Dropout(.1),nn.Linear(n,n*2),nn.GLU());self.att_norm=nn.LayerNorm(n)
        self.ff=GRN(n);self.final_gate=nn.Sequential(nn.Dropout(.1),nn.Linear(n,n*2),nn.GLU());self.final_norm=nn.LayerNorm(n)
    def forward(self,history,batch):
        B=history.shape[0]; c,h,c0,e=[g(self.neutral.expand(B,-1)) for g in self.contexts]
        x=torch.cat([history,batch['history_calendar']],-1) if self.bundle in ('X1','X3') else history
        past=self.history_selection(x,c)
        future=self.position.weight[None].expand(B,-1,-1)
        if self.future_selection is not None:future=future+self.future_selection(batch['future_calendar'],c)
        if self.past_calendar is not None:future=future+self.past_calendar(batch['history_calendar'].flatten(1))[:,None,:]
        p,state=self.encoder(past,(h[None].contiguous(),c0[None].contiguous()))
        f,_=self.decoder(future,state)
        local=self.local_norm(torch.cat([past,future],1)+self.local_gate(torch.cat([p,f],1)))
        z=self.enrich(local,e); length=z.shape[1]; d=z.shape[-1]//self.heads
        q=self.q(z).reshape(B,length,self.heads,d).transpose(1,2)
        k=self.k(z).reshape(B,length,self.heads,d).transpose(1,2)
        scores=(q@k.transpose(-2,-1))/math.sqrt(d)
        mask=torch.ones(length,length,device=z.device,dtype=torch.bool).triu(1)
        a=scores.masked_fill(mask,float('-inf')).softmax(-1)
        self.last_attention=a.detach().mean(1)
        attended=(a@self.v(z)[:,None]).mean(1)
        z=self.att_norm(z+self.att_gate(self.out(attended)))
        z=self.final_norm(local+self.final_gate(self.ff(z)))
        return z[:,-self.horizon:]
