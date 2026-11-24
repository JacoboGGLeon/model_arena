"""Curvas de entrenamiento sin leyendas masivas ni mezcla de folds."""
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def loss_figure(curves,highlight=None):
    folds=[f for f in ['F1','F2','F3','FUTURO'] if f in set(curves.fold)]
    if not folds:return go.Figure()
    fig=make_subplots(rows=2,cols=len(folds),shared_xaxes='columns',horizontal_spacing=.055,vertical_spacing=.16,
        subplot_titles=[target+' · '+fold for target in ['Saldo','Variación'] for fold in folds])
    for row,target in [(1,'saldo'),(2,'variacion')]:
        for col,fold in enumerate(folds,1):
            frame=curves[(curves.target==target)&(curves.fold==fold)]
            # Dibujar el resaltado al final para que quede delante.
            candidates=sorted(frame.candidate.unique(),key=lambda name:(name==highlight,name))
            for candidate in candidates:
                g=frame[frame.candidate==candidate].sort_values('epoch')
                active=candidate==highlight
                fig.add_trace(go.Scatter(x=g.epoch,y=g.train_loss,name=candidate,mode='lines',showlegend=False,
                    line={'color':('#1464a5' if row==1 else '#02a5a5') if active else '#a8b5c5','width':3 if active else 1},opacity=1 if active else .45,
                    hovertemplate=candidate+' · '+fold+'<br>Época %{x}<br>Loss %{y:.6g}<extra></extra>'),row=row,col=col)
            fig.update_xaxes(title_text='Época' if row==2 else None,row=row,col=col)
            fig.update_yaxes(title_text='Pinball' if col==1 else None,row=row,col=col)
    fig.update_layout(template='plotly_white',height=650,margin={'t':65,'b':50,'l':55,'r':20},hovermode='closest')
    return fig
