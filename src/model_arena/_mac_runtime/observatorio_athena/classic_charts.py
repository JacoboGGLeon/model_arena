import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.colors import qualitative

def figure(curves):
    folds=[f for f in ['F1','F2','F3','FUTURO'] if f in set(curves.fold)]
    if not folds:return go.Figure()
    fig=make_subplots(rows=2,cols=len(folds),subplot_titles=[t+' · '+f for t in ['Saldo','Variación'] for f in folds],shared_xaxes='columns',vertical_spacing=.17)
    names=sorted(curves.candidate.unique());colors={n:qualitative.Plotly[i%10] for i,n in enumerate(names)}
    for row,target in [(1,'saldo'),(2,'variacion')]:
        for col,fold in enumerate(folds,1):
            data=curves[(curves.target==target)&(curves.fold==fold)]
            for (candidate,split),g in data.groupby(['candidate','split']):
                g=g.sort_values('epoch');label=candidate+' · '+('Entrenamiento' if split=='train' else 'Validación')
                fig.add_trace(go.Scatter(x=g.epoch,y=g.loss,name=label,legendgroup=label,showlegend=row==1 and col==1,line={'color':colors[candidate],'dash':'solid' if split=='train' else 'dash'},hovertemplate=label+'<br>Época %{x}<br>Loss %{y:.9g}<extra></extra>'),row=row,col=col)
            fig.update_xaxes(title_text='Época' if row==2 else None,row=row,col=col)
            fig.update_yaxes(title_text='Pinball escalado' if col==1 else None,row=row,col=col)
    fig.update_layout(template='plotly_white',height=700,legend={'orientation':'h','y':-.17},margin={'t':65,'b':100})
    return fig
