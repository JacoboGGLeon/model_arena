"""Vista C2.1 de alertamientos sobre consultas publicadas en Athena."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from .alert_taxonomy import SEVERITY_COLORS
from .common import literal

FOLDS = ['F1', 'F2', 'F3', 'FUTURO']
GRADES = ['MUY BAJO', 'BAJO', 'MEDIO', 'ALTO']
COLORS = {'Observado': '#235a97', 'Modelo': '#007a71', 'Baseline': '#ba650f'}


def visible_frames(history, daily, events, stage, context):
    history, daily, events = history.copy(), daily.copy(), events.copy()
    for frame in (history, daily, events):
        frame['date'] = pd.to_datetime(frame['date'])
    if stage == 'Historia previa':
        limit = daily.date.min()
        return history[history.date < limit].groupby('target', group_keys=False).tail(context), daily.iloc[:0], events.iloc[:0]
    if stage == 'Tres folds':
        daily = daily[daily.fold.isin(FOLDS[:3])]
    elif stage != 'Todos':
        daily = daily[daily.fold == stage]
    if daily.empty:
        return history.iloc[:0], daily, events.iloc[:0]
    first, last = daily.date.min(), daily.date.max()
    prior = history[history.date < first].groupby('target', group_keys=False).tail(context)
    start = prior.date.min() if len(prior) else first
    history = history[history.date.between(start, last)]
    events = events[events.date.between(first, last) & events.fold.isin(daily.fold.unique())]
    return history, daily, events


def alert_figure(history, daily, events, families, bands, *, scale='Comprimida · admite negativos', detail=False, mad=False):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[.43, .43, .14],
                        vertical_spacing=.065, subplot_titles=['Saldo', 'Variación', 'Timeline'])
    for row, target in [(1, 'saldo'), (2, 'variacion')]:
        h = history[history.target == target].sort_values('date')
        d = daily[daily.target == target].sort_values('date')
        observed = pd.to_numeric(h.observed, errors='coerce').dropna().to_numpy()
        amplitude = max(float(np.quantile(np.abs(observed), .75)), 1.) if len(observed) else 1.
        def transform(values):
            values = np.asarray(values, dtype=float)
            return np.arcsinh(values / amplitude) if scale.startswith('Comprimida') else values
        sources = [('Observado', 'hist', h)] + [(family, prefix, g) for family, prefix in [('Modelo','model'),('Baseline','baseline')] for _,g in d.groupby('fold',sort=False)]
        legends = set()
        for family, prefix, frame in sources:
            if family not in families or frame.empty:
                continue
            color = COLORS[family]
            if isinstance(bands, bool):
                show_band = bands
            else:
                show_band = family in bands
            lower, upper = prefix+'_q025', prefix+'_q975'
            if show_band and lower in frame and upper in frame:
                fill = 'rgba('+','.join(str(int(color[i:i+2],16)) for i in (1,3,5))+',.12)'
                fig.add_trace(go.Scatter(x=frame.date,y=transform(frame[lower]),line={'width':0},showlegend=False,hoverinfo='skip',connectgaps=False),row=row,col=1)
                fig.add_trace(go.Scatter(x=frame.date,y=transform(frame[upper]),line={'width':0},fill='tonexty',fillcolor=fill,showlegend=False,hoverinfo='skip',connectgaps=False),row=row,col=1)
            curves = [(prefix+'_q50','Puntual' if mad and family!='Observado' else 'Q50')]
            if detail:
                curves = [(lower,'Inferior' if mad and family!='Observado' else 'Q2.5'),
                          (prefix+'_q50','Puntual' if mad and family!='Observado' else 'Q50'),
                          (upper,'Superior' if mad and family!='Observado' else 'Q97.5')]
            if family == 'Observado':
                curves.append(('observed','Real'))
            for column, label in curves:
                if column not in frame or (family == 'Observado' and label != 'Real' and not detail):
                    continue
                name = family+' · '+label
                fig.add_trace(go.Scatter(x=frame.date,y=transform(frame[column]),customdata=pd.to_numeric(frame[column],errors='coerce'),
                    name=name,legendgroup=name,showlegend=row==1 and name not in legends,
                    line={'color':color,'width':2 if label=='Real' else 1.3,'dash':'solid' if label=='Real' else {'Q2.5':'dot','Q50':'dash','Q97.5':'longdash','Inferior':'dot','Puntual':'dash','Superior':'longdash'}[label]},
                    connectgaps=False,hovertemplate='%{x}<br>'+name+' %{customdata:,.2f}<extra></extra>'),row=row,col=1)
                legends.add(name)
        e = events[events.target == target]
        if len(e):
            alerts = e[e.has_alert == True]
            for direction, symbol in [('INFERIOR','triangle-down'),('SUPERIOR','triangle-up'),('OTRA','square')]:
                reasons = alerts.reasons.fillna('')
                mask = ~reasons.str.contains('INFERIOR|SUPERIOR') if direction=='OTRA' else reasons.str.contains(direction)
                selected = alerts[mask]
                if selected.empty:
                    continue
                labels = selected.reasons.fillna('')+' · '+selected.severity.fillna('')
                fig.add_trace(go.Scatter(x=selected.date,y=transform(selected.observed),customdata=selected.observed,
                    text=labels,mode='markers',name='Alertas · '+target,showlegend=False,
                    marker={'symbol':symbol,'size':9,'color':[SEVERITY_COLORS.get(v,'#607d8b') for v in selected.severity]},
                    hovertemplate='%{x}<br>%{text}<br>Real %{customdata:,.2f}<extra></extra>'),row=row,col=1)
            if target=='saldo':
                overdraft=e[e.overdraft == True]
                fig.add_trace(go.Scatter(x=overdraft.date,y=transform(overdraft.observed),customdata=overdraft.observed,
                    mode='markers',name='◆ Sobregiro',marker={'color':'#8739b5','symbol':'diamond-open','size':14},
                    hovertemplate='%{x}<br>Sobregiro %{customdata:,.2f}<extra></extra>'),row=row,col=1)
            points=e[(e.has_alert == True) | (e.overdraft == True)]
            labels=points.reasons.fillna('')+' · '+points.severity.fillna('')
            fig.add_trace(go.Scatter(x=points.date,y=[0]*len(points),mode='markers',name='Timeline · '+target,
                text=labels,marker={'color':[SEVERITY_COLORS.get(v,'#607d8b') if target=='saldo' else '#15202b' for v in points.severity],
                'symbol':'circle' if target=='saldo' else 'circle-open','size':9 if target=='saldo' else 17},
                hovertemplate='%{x}<br>'+target+' · %{text}<extra></extra>'),row=3,col=1)
        if scale=='Acercar al observado' and len(observed):
            low, high = float(observed.min()), float(observed.max())
            margin=max((high-low)*.15, abs(high)*.02, 1.)
            fig.update_yaxes(range=[low-margin, high+margin],row=row,col=1)
        fig.update_yaxes(title_text='Importe ($)', tickprefix='$ ', row=row,col=1)
        if scale.startswith('Comprimida'):
            ys=np.concatenate([np.asarray(t.y,dtype=float) for t in fig.data if t.yaxis==('y' if row==1 else 'y2') and t.y is not None]) if fig.data else np.array([0.])
            ys=ys[np.isfinite(ys)]
            if len(ys):
                ticks=np.linspace(min(ys.min(),0.),max(ys.max(),0.),7)
                fig.update_yaxes(tickmode='array',tickvals=ticks,ticktext=['$ '+format(v,',.0f') for v in np.sinh(ticks)*amplitude],row=row,col=1)
    for fold, group in daily.groupby('fold'):
        first=pd.to_datetime(group.date).min()
        fig.add_vline(x=first,line_dash='dot',line_color='#b4c4d3',row='all',col=1)
        fig.add_annotation(x=first,y=1.04,xref='x',yref='paper',text=fold,showarrow=False)
    fig.update_yaxes(range=[-1,1],showticklabels=False,showgrid=False,zeroline=True,row=3,col=1)
    fig.update_xaxes(title_text='Fecha',row=3,col=1)
    fig.update_xaxes(matches='x3',showspikes=True,spikemode='across',spikesnap='cursor',gridcolor='#eef2f6')
    fig.update_layout(template='plotly_white',height=720,hovermode='x unified',margin={'t':60,'l':80,'r':20,'b':95},
                      legend={'orientation':'h','y':-.14},uirevision='c21-'+scale)
    return fig


def render_alerts(repo, data, paged, filters):
    a,b,c = st.columns(3)
    origin=a.selectbox('Origen del pronóstico',['Winner','Baseline'])
    if repo.mac2_events:
        st.caption('Motor de reglas: MAC 2 / Engine 3 · Intervalo: envolvente MAD.')
    block=b.selectbox('Tramo del conteo',['Todos','Tres folds',*FOLDS])
    objective=c.selectbox('Objetivo de la alerta',['Cualquiera de los dos','saldo','variacion','Ambos el mismo día'])
    a,b = st.columns(2)
    severity=a.selectbox('Severidad',['Todas',*GRADES,'SIN_GRADUAR'])
    cause=b.selectbox('Causa',['Todas','Alertamiento','Sobregiro'])
    args=dict(filters=filters,source=origin,include_rules=True,fold=block if block in FOLDS else None,
              target=None if objective=='Cualquiera de los dos' else objective,
              severity=None if severity=='Todas' else severity,cause=None if cause=='Todas' else cause)
    sql=repo.events(**args)
    if block=='Tres folds':
        sql='SELECT * FROM ('+sql+") e WHERE fold IN ('F1','F2','F3')"
    dates=data('SELECT DISTINCT date FROM ('+sql+') e ORDER BY date')
    date=st.selectbox('Fecha del conteo',['Todas',*dates.date.astype(str).tolist()])
    if date!='Todas':
        sql='SELECT * FROM ('+sql+') e WHERE date=DATE '+literal(date)
    counts=data("SELECT count(DISTINCT CASE WHEN has_alert AND severity IN ('MUY BAJO','BAJO','MEDIO','ALTO') THEN account_id END) AS alert_accounts,count(DISTINCT CASE WHEN overdraft THEN account_id END) AS overdraft_accounts FROM ("+sql+') e').iloc[0]
    a,b=st.columns(2)
    a.metric('Cuentas con alerta por severidad',int(counts.alert_accounts))
    b.metric('◆ Cuentas con sobregiro',int(counts.overdraft_accounts))
    st.caption('Conteos independientes de cuentas únicas en los filtros. Las cuentas cero aplican sus umbrales contables en ambos enfoques. Los cruces SIN_GRADUAR quedan disponibles en el detalle.')
    ranks="CASE WHEN has_alert THEN CASE severity WHEN 'MUY BAJO' THEN 1 WHEN 'BAJO' THEN 2 WHEN 'MEDIO' THEN 3 WHEN 'ALTO' THEN 4 ELSE 0 END ELSE 0 END"
    grades=data('SELECT severity_rank,count(*) AS accounts FROM (SELECT account_id,max('+ranks+') AS severity_rank FROM ('+sql+') e GROUP BY account_id) g WHERE severity_rank>0 GROUP BY severity_rank ORDER BY severity_rank')
    names=[GRADES[int(value)-1] for value in grades.severity_rank]
    fig=go.Figure(go.Bar(x=names,y=grades.accounts,marker_color=[SEVERITY_COLORS[v] for v in names],text=grades.accounts,textposition='outside'))
    fig.update_layout(template='plotly_white',height=230,margin={'t':20,'b':35},yaxis_title='Cuentas contables')
    st.plotly_chart(fig,use_container_width=True)
    only=st.checkbox('Explorar sólo cuentas con alertas en estos filtros',False)
    accounts_sql='SELECT account_id,currency,budget,taxonomy,max(CASE WHEN has_alert THEN 1 ELSE 0 END) AS has_alert,max(CASE WHEN overdraft THEN 1 ELSE 0 END) AS overdraft FROM ('+sql+') e'+(' WHERE has_alert OR overdraft' if only else '')+' GROUP BY account_id,currency,budget,taxonomy ORDER BY account_id'
    def decorate_accounts(frame):
        import base64
        from .account_previews import spark
        ids=','.join(literal(x) for x in frame.account_id)
        history_preview=data('SELECT account_id,date,target,observed FROM '+repo.table('history')+' WHERE '+repo.where()+' AND account_id IN ('+ids+') ORDER BY account_id,date,target')
        marked=data('SELECT account_id,date,target,has_alert,overdraft FROM ('+sql+') e WHERE account_id IN ('+ids+')')
        for target,label,color in [('saldo','Saldo','#1464a5'),('variacion','Variación','#02a5a5')]:
            pictures={}
            for account_id in frame.account_id:
                g=history_preview[(history_preview.account_id==account_id)&(history_preview.target==target)].drop_duplicates('date').sort_values('date')
                e=marked[(marked.account_id==account_id)&(marked.target==target)&(marked.has_alert|marked.overdraft)]
                pic=spark(g.date.tolist(),g.observed.to_numpy(),set(e.date),color,target)
                pictures[account_id]='data:image/png;base64,'+base64.b64encode(pic).decode()
            frame[label]=frame.account_id.map(pictures)
        return frame
    with st.expander('Cuentas del universo filtrado',expanded=True):
        accounts=paged(accounts_sql,'alerts',decorate=decorate_accounts)
        st.caption('Filas y miniaturas · historia completa. Azul: saldo; turquesa: variación. Punto rojo: evento de saldo; aro negro: evento de variación. Los marcadores responden al origen, tramo, objetivo, severidad, causa y fecha del conteo.')
    if accounts.empty:
        st.info('Sin cuentas para los filtros seleccionados.')
        return
    if st.session_state.get('alerts_account') not in accounts.account_id.tolist():st.session_state['alerts_account']=accounts.account_id.iloc[0]
    account=st.selectbox('Cuenta de la página',accounts.account_id.tolist(),key='alerts_account')
    history=data(repo.history(account))
    daily=data(repo.daily(account))
    # El tramo del gráfico es independiente del tramo utilizado por los conteos.
    event_args={**args,'fold':None,'account':account}
    events=data(repo.events(**event_args))
    st.subheader('Detalle de la cuenta · '+account)
    routing=daily[['target','fold','candidate','input']].drop_duplicates()
    final=routing[routing.fold=='FUTURO']
    if repo.mac2_events:
        st.caption('MAC 2 · cada tramo muestra su envolvente MAD publicada. Los marcadores del saldo y la variación confirmados comparten la severidad del saldo que emite Engine 3.')
    elif final.empty:
        st.warning('No se publicó la asignación final de esta serie. No se mostrará una composición como ganador.')
        daily=daily.iloc[:0]
    else:
        winners=final.drop_duplicates('target').set_index('target')['candidate'].to_dict()
        missing=routing.loc[routing.candidate!=routing.target.map(winners),'fold'].unique().tolist()
        st.success(' · '.join(str(t)+': '+str(c) for t,c in winners.items()))
        daily=daily.copy()
        daily.loc[daily.candidate!=daily.target.map(winners),['model_q025','model_q50','model_q975']]=np.nan
        if missing:st.info('Sin curva histórica publicada del ganador en '+', '.join(missing)+'. Se conserva Baseline y Observado; no se reconstruye el pasado con pesos finales.')
    with st.expander('Modelo asignado y procedencia'):
        if not final.empty:
            st.dataframe(final[['target','candidate','input']].drop_duplicates().rename(columns={'candidate':'Winner de la serie','input':'Input'}),hide_index=True,use_container_width=True)
            st.caption('Asignación final usada para esta serie. Los tramos históricos disponibles se indican arriba.')
        else:st.info('Sin ganador final publicado.')
    a,b=st.columns(2)
    stage=a.selectbox('Tramo visible en la gráfica',['Todos','Tres folds','Historia previa',*FOLDS])
    context=int(b.number_input('Cierres de contexto anterior',min_value=0,max_value=501,value=20,step=5))
    a,b=st.columns(2)
    preset=a.radio('Detalle',['Lectura rápida','Detalle 3 × 3'],horizontal=True)
    scale=b.radio('Escala',['Lineal','Comprimida · admite negativos','Acercar al observado'],index=1)
    families=st.pills('Familias visibles',['Baseline','Winner','Observado'],selection_mode='multi',default=['Observado','Winner'],key='c21_family_buttons')
    families=['Modelo' if v=='Winner' else v for v in families]
    bands=st.pills('Bandas de referencia' if repo.mac2_events else 'Bandas Q2.5–Q97.5',['Baseline','Winner','Observado'],selection_mode='multi',default=['Observado','Winner'],key='c21_band_buttons')
    bands=['Modelo' if v=='Winner' else v for v in bands]
    h,d,e=visible_frames(history,daily,events,stage,context)
    st.caption('Alertamientos · '+origin+' · Las bandas observadas describen la historia anterior. Ambos ejes monetarios muestran $; la compresión cambia el espaciado, no los montos.' + ('' if repo.mac2_events else ' Winner usa sólo el candidato final por serie.'))
    if h.empty and d.empty:
        st.info('Sin observaciones para el tramo seleccionado.')
        return
    st.plotly_chart(alert_figure(h,d,e,families,bands,scale=scale,detail=preset=='Detalle 3 × 3',mad=repo.mac2_events),use_container_width=True)
    st.caption('Timeline · Punto para saldo, aro para variación y ambos superpuestos cuando coinciden. El rombo morado marca sobregiro. Los marcadores se anclan al importe observado de su fecha.')
    a,b=st.columns(2)
    for target,column in [('saldo',a),('variacion',b)]:
        observed=d[(d.target==target) & d.observed.notna()].sort_values('date')
        active=set(pd.to_datetime(e[(e.target==target) & e.has_alert & e.severity.isin(GRADES)].date))
        streak=0
        for current in reversed(pd.to_datetime(observed.date).tolist()):
            if current not in active:break
            streak+=1
        column.metric('Racha actual · '+target,str(streak)+' cierres')
    with st.expander('Fechas y motivos'):
        st.dataframe(e,hide_index=True,use_container_width=True)
    with st.expander('Valores de las curvas'):
        st.dataframe(d,hide_index=True,use_container_width=True)
    st.download_button('Descargar eventos de la cuenta',e.to_csv(index=False),file_name='eventos_'+account+'.csv')

    with st.expander('Salidas MAC · descarga por cuenta'):
        st.caption('Engine 2: proyecciones y métricas en layout MAC. Engine 3: validación y alertamiento en layout MAC. La descarga conserva las columnas publicadas y su procedencia; los archivos publicados pueden contener asignaciones históricas por fold.')
        for engine in ['engine2','engine3']:
            if st.button('Preparar '+engine.upper()+' de esta cuenta',key='prepare_'+engine):
                from .schemas import ENGINE2_COLUMNS
                from .mac_rules_reference import OFFICIAL_COLUMNS
                names=ENGINE2_COLUMNS if engine=='engine2' else OFFICIAL_COLUMNS
                export=data('SELECT '+','.join(names)+' FROM '+repo.table(engine)+' WHERE '+repo.where(['cross_key_id='+literal(account)]))
                st.download_button('Descargar '+engine.upper()+' CSV',export.to_csv(index=False),file_name=engine+'_'+account+'.csv',key='download_'+engine)
