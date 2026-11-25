from dataclasses import asdict
import hashlib,json
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from .repository import Repository,Filters
from model_arena.presentation import publication_view, winner_rows
from .layout_export import write_published_csv
from .theme import CSS
from .analysis import comparisons,score
from .historical_sigma import SIGMA_BANDS
from .alert_taxonomy import SEVERITY_COLORS


def history_figure(history,bands=True):
 fig=make_subplots(rows=2,cols=1,shared_xaxes=True,subplot_titles=['Saldo','Variación'],vertical_spacing=.15)
 for row,target,color in [(1,'saldo','#1464a5'),(2,'variacion','#02a5a5')]:
  g=history[history.target==target].sort_values('date')
  if bands:
   seen=set()
   for low,high,label,shade,alpha in SIGMA_BANDS:
    lo=g.sigma_center+low*g.sigma_std;hi=g.sigma_center+high*g.sigma_std
    fill='rgba('+','.join(str(int(shade[i:i+2],16)) for i in (1,3,5))+','+str(alpha)+')'
    fig.add_trace(go.Scatter(x=g.date,y=lo,line={'width':0},showlegend=False,hoverinfo='skip',connectgaps=False),row=row,col=1)
    fig.add_trace(go.Scatter(x=g.date,y=hi,line={'width':0},fill='tonexty',fillcolor=fill,name='Referencia '+label,showlegend=row==1 and label not in seen,connectgaps=False),row=row,col=1);seen.add(label)
  fig.add_trace(go.Scatter(x=g.date,y=g.observed,name=target.capitalize(),line={'color':color},customdata=g[['zscore','sigma_reference_n','zscore_status']].to_numpy(),hovertemplate='%{x}<br>Observado %{y:,.2f}<br>Z %{customdata[0]} · n %{customdata[1]}<br>%{customdata[2]}<extra></extra>'),row=row,col=1)
  fig.add_trace(go.Scatter(x=g.date,y=g.sigma_center,name='Media anterior',line={'color':'#8091a5','dash':'dot'},showlegend=row==1),row=row,col=1)
 fig.update_layout(template='plotly_white',height=680,hovermode='x unified',legend={'orientation':'h','y':-.12})
 return fig


def alert_figure(history,daily,events,families,bands):
 fig=make_subplots(rows=3,cols=1,shared_xaxes=True,row_heights=[.43,.43,.14],vertical_spacing=.09,subplot_titles=['Saldo','Variación','Timeline'])
 for row,target in [(1,'saldo'),(2,'variacion')]:
  h=history[history.target==target].sort_values('date');d=daily[daily.target==target].sort_values('date')
  if 'Observado' in families:fig.add_trace(go.Scatter(x=h.date,y=h.observed,name='Observado',line={'color':'#235a97'},showlegend=row==1),row=row,col=1)
  for family,prefix,color in [('Winner','model','#007a71'),('Baseline','baseline','#ba650f')]:
   if family not in families:continue
   for index,(fold,g) in enumerate(d.groupby('fold',sort=False)):
    if bands:
     fig.add_trace(go.Scatter(x=g.date,y=g[prefix+'_q025'],line={'width':0},showlegend=False,hoverinfo='skip'),row=row,col=1)
     fig.add_trace(go.Scatter(x=g.date,y=g[prefix+'_q975'],line={'width':0},fill='tonexty',fillcolor='rgba('+','.join(str(int(color[i:i+2],16)) for i in (1,3,5))+',.12)',showlegend=False,hoverinfo='skip'),row=row,col=1)
    fig.add_trace(go.Scatter(x=g.date,y=g[prefix+'_q50'],name=family+' · Q50',line={'color':color,'dash':'dash' if family=='Winner' else 'dot'},showlegend=row==1 and index==0,customdata=[fold]*len(g),hovertemplate='%{x}<br>%{customdata}<br>%{y:,.2f}<extra></extra>'),row=row,col=1)
  if not events.empty:
   for e in events[events.target==target].to_dict('records'):
    label=e['reasons']+' · '+e['severity'];color=SEVERITY_COLORS.get(e['severity'],'#607d8b')
    if e['has_alert']:fig.add_trace(go.Scatter(x=[e['date']],y=[e['observed']],mode='markers',name=label,text=[label],marker={'color':color,'symbol':'triangle-down' if 'INFERIOR' in e['reasons'] else 'triangle-up','size':10},showlegend=False,hovertemplate='%{x}<br>%{text}<br>%{y:,.2f}<extra></extra>'),row=row,col=1)
    if e['overdraft']:fig.add_trace(go.Scatter(x=[e['date']],y=[e['observed']],mode='markers',name='Sobregiro',marker={'color':'#8739b5','symbol':'diamond-open','size':14},showlegend=False),row=row,col=1)
    fig.add_trace(go.Scatter(x=[e['date']],y=[0],mode='markers',name=label,text=[target+' · '+label],marker={'color':color,'symbol':'circle' if target=='saldo' else 'circle-open','size':9 if target=='saldo' else 15},showlegend=False,hovertemplate='%{x}<br>%{text}<extra></extra>'),row=3,col=1)
 for fold,g in daily.groupby('fold'):
  start=g.date.min();fig.add_vline(x=start,line_dash='dot',line_color='#b4c4d3')
  fig.add_annotation(x=start,y=1.08,xref='x',yref='paper',text=fold,showarrow=False)
 fig.update_yaxes(range=[-1,1],showticklabels=False,row=3,col=1)
 fig.update_xaxes(matches='x3',showspikes=True,spikemode='across')
 fig.update_layout(template='plotly_white',height=740,hovermode='x unified',legend={'orientation':'h','y':-.12})
 return fig


def render_app(config,athena):
 st.set_page_config(page_title='Observatorio de Cuentas Contables',layout='wide')
 st.markdown(CSS,unsafe_allow_html=True)
 st.sidebar.markdown('<div class="obs-brand">Observatorio</div><div class="obs-brand-sub">CUENTAS CONTABLES · STREAMLIT + ATHENA · 0.13.1</div>',unsafe_allow_html=True)
 tag=hashlib.sha256(json.dumps(asdict(config),sort_keys=True).encode()).hexdigest()
 @st.cache_data(ttl=300,max_entries=128,show_spinner=False)
 def fetch(sql,cache_tag,_client):return _client.all(sql)[0]
 @st.cache_data(ttl=300,max_entries=128,show_spinner=False)
 def result_page(handle,token,cache_tag,_client):return _client.page(handle,token)
 def data(sql):
  with st.spinner('Consultando resultados…'):return fetch(sql,tag,athena)
 def paged(sql,key,decorate=None):
  statekey='query_'+key
  if statekey not in st.session_state or st.session_state[statekey]['sql']!=sql:
   with st.spinner('Preparando consulta…'):handle=athena.execute(sql)
   st.session_state[statekey]=dict(sql=sql,handle=handle,tokens=[None],page=0)
  state=st.session_state[statekey];index=state['page']
  frame,next_token=result_page(state['handle'],state['tokens'][index],tag,athena)
  displayed=decorate(frame.copy()) if decorate is not None and not frame.empty else frame
  display=displayed.rename(columns={'account_id':'Cuenta','currency':'Divisa','taxonomy':'Taxonomía','budget':'Budget','cohort':'Población','observations':'Observaciones','candidate':'Modelo asignado','selection_source':'Origen de selección','has_alert':'Alertamiento','overdraft':'Sobregiro'})
  image_columns={name:st.column_config.ImageColumn(name,width='medium') for name in ('Saldo','Variación') if name in display}
  if image_columns:
   selection=st.dataframe(display,hide_index=True,use_container_width=True,column_config=image_columns,row_height=52,selection_mode='single-row',on_select='rerun',key=key+'_rows_'+str(index)+'_'+hashlib.sha256(sql.encode()).hexdigest()[:12])
   if selection.selection.rows:st.session_state['alerts_account']=frame.account_id.iloc[selection.selection.rows[0]]
  else:st.dataframe(display,hide_index=True,use_container_width=True)
  left,middle,right=st.columns([1,3,1]);middle.caption(f'Página {index+1} · {len(frame)} filas · Lectura Athena: {state["handle"].get("bytes_scanned",0)/1024**2:.2f} MB')
  if left.button('Anterior',disabled=index==0,key=key+'_prev'):state['page']-=1;st.rerun()
  if right.button('Siguiente',disabled=not next_token,key=key+'_next'):
   if len(state['tokens'])<=index+1:state['tokens'].append(next_token)
   state['page']+=1;st.rerun()
  return frame
 try:pubs=data(f'SELECT * FROM {config.table("publications")} ORDER BY published_at DESC,publication_id DESC')
 except Exception as exc:st.error('No se pudo consultar el registro de publicaciones. '+str(exc));return
 if pubs.empty:st.info('Publica primero una corrida con 10_Slow_System.ipynb.');return
 options=pubs.publication_id.tolist()
 initial=options.index(config.initial_publication_id) if config.initial_publication_id in options else 0
 pub=st.sidebar.selectbox('Corrida publicada',options,index=initial,format_func=lambda p:pubs.loc[pubs.publication_id==p,'run_id'].iloc[0][13:]+' · '+pd.to_datetime(pubs.loc[pubs.publication_id==p,'published_at'].iloc[0],utc=True).tz_convert(config.display_timezone).strftime('%Y-%m-%d %H:%M'))
 selected=pubs[pubs.publication_id==pub].iloc[0]
 repo=Repository(config,selected.run_id,pub,int(selected.shards))
 @st.cache_data(ttl=300,max_entries=32,show_spinner=False)
 def read_view(uri,cache_tag,_client):
  return _client.publication_manifest(uri).get('view')
 view=publication_view(['saldo','variacion'],[.025,.5,.975],labels={'saldo':'Saldo','variacion':'Variación'},
                       metrics=['SMAPE','MAE','RMSE','Coverage'],routing=['account_id','target'])
 try:
  view=read_view(selected.manifest_uri,tag,athena) or view
 except Exception:
  st.sidebar.caption('Se usan las etiquetas MAC de compatibilidad para esta publicación.')
 targets={item['id']:item for item in view['targets']}
 page=st.sidebar.radio('Monitor',['Historia y budgets','Model Arena','Alertamientos'])
 if st.sidebar.button('Actualizar resultados'):
  fetch.clear();result_page.clear();read_view.clear()
  for k in list(st.session_state):
   if k.startswith('query_'):del st.session_state[k]
  st.rerun()
 with st.sidebar.expander('Procedencia'):
  st.caption('Publicación: '+pub);st.caption(selected.protocol);st.caption('Versión del monitor: Streamlit Athena 0.12.1')
  if st.button('Ver calendario de esta publicación',key='calendar_provenance'):
   try:
    calendar_info=data(f'SELECT * FROM {repo.table("calendar_provenance")} WHERE {repo.where()}')
    if calendar_info.empty:st.caption('La publicación no incluye procedencia del calendario.')
    else:
     details=calendar_info.iloc[0]
     st.caption('Generador: '+str(details.generator)+' · '+str(details.generator_version))
     st.caption('Perfil: '+str(details.profile)+' · '+str(details.feature_count)+' variables')
     st.caption(str(details.start_date)+' → '+str(details.end_date))
     st.caption('SHA-256: '+str(details.calendar_sha256))
     st.json(json.loads(details.metadata_json))
   except Exception as exc:
    if 'TABLE_NOT_FOUND' in str(exc):st.caption('Esta publicación anterior no incluye procedencia del calendario.')
    else:st.warning('No se pudo consultar la procedencia del calendario: '+str(exc))
 dimensions=data(f'SELECT currency,taxonomy,budget,cohort,count(*) AS accounts FROM {repo.table("inventory")} WHERE {repo.where()} GROUP BY currency,taxonomy,budget,cohort')
 currencies=sorted(dimensions.currency.dropna().unique());taxonomies=sorted(dimensions.taxonomy.dropna().unique());budgets=sorted(int(x) for x in dimensions.budget.dropna().unique())
 with st.expander('Descargar salidas MAC publicadas'):
  st.caption('Engine 2: proyecciones y métricas. Engine 3: validación y alertamiento. Layout original publicado, con sus asignaciones históricas; descarga independiente de los filtros de pantalla.')
  engine=st.selectbox('Salida',['engine2','engine3'])
  delivery_fold=st.selectbox('Tramo de la salida',['F1','F2','F3','FUTURO'])
  if st.button('Preparar descarga MAC'):
   try:
    with st.spinner('Preparando el CSV completo…'), TemporaryDirectory(prefix='mac-export-') as directory:
     path=Path(directory)/(engine+'_'+delivery_fold+'.csv')
     with path.open('w',encoding='utf-8',newline='') as stream:
      count=write_published_csv(athena,repo,engine,delivery_fold,stream)
     if not count:st.info('Esta salida no tiene filas para el tramo seleccionado.')
     else:st.caption(f'{count:,} filas disponibles para descargar.')
     with path.open('rb') as stream:
      st.download_button('Descargar salida MAC CSV',stream,file_name=path.name,mime='text/csv',on_click='ignore')
   except Exception as exc:
    st.error('No se pudo preparar la descarga MAC: '+str(exc))
 st.title(page)
 with st.form('filters_'+page):
  a,b,c=st.columns(3)
  currency=a.selectbox('Divisa',['Todas']+currencies,index=(['Todas']+currencies).index('MXP') if 'MXP' in currencies else 0)
  taxonomy=b.selectbox('Taxonomía',['Todas']+taxonomies)
  budget=c.selectbox('Budget',['Todos']+budgets)
  search=st.text_input('Buscar cuenta') if page!='Model Arena' else ''
  st.form_submit_button('Aplicar filtros')
 filters=Filters(currency=None if currency=='Todas' else currency,taxonomy=None if taxonomy=='Todas' else taxonomy,budget=None if budget=='Todos' else budget,search=search)
 if page=='Historia y budgets':
  summary=data(repo.summary(filters));total=int(summary.accounts.sum()) if not summary.empty else 0
  st.metric('Cuentas contables',f'{total:,}')
  summary_view=st.radio('Vista del resumen',['Gráfico','Tabla'],horizontal=True,key='history_summary_view')
  tax=summary.groupby('taxonomy',dropna=False).accounts.sum().reset_index()
  matrix=summary.assign(budget=summary.budget.map(lambda v:'Sin budget' if pd.isna(v) else 'B'+str(int(v)))).pivot_table(index='taxonomy',columns='budget',values='accounts',aggfunc='sum',fill_value=0)
  matrix=matrix.reindex(columns=[f'B{i}' for i in range(1,11)]+(['Sin budget'] if 'Sin budget' in matrix.columns else []),fill_value=0)
  a,b=st.columns(2)
  with a:
   st.subheader('Cuentas por taxonomía')
   if summary_view=='Tabla':st.dataframe(tax,hide_index=True,use_container_width=True)
   else:
    fig=go.Figure(go.Bar(y=tax.taxonomy,x=tax.accounts,orientation='h',text=tax.accounts,textposition='auto',marker_color='#1464a5'))
    fig.update_layout(template='plotly_white',height=350,xaxis_title='Cuentas',margin={'t':15,'b':45,'l':10,'r':20})
    st.plotly_chart(fig,use_container_width=True)
  with b:
   st.subheader('Taxonomía × budget')
   if summary_view=='Tabla':st.dataframe(matrix,use_container_width=True)
   else:
    fig=go.Figure(go.Heatmap(x=matrix.columns,y=matrix.index,z=matrix.to_numpy(),colorscale='Blues',texttemplate='%{z:,.0f}',hovertemplate='%{y} · %{x}<br>%{z:,.0f} cuentas<extra></extra>'))
    fig.update_layout(template='plotly_white',height=350,xaxis_title='Budget',margin={'t':15,'b':45,'l':10,'r':20})
    st.plotly_chart(fig,use_container_width=True)
  a,b=st.columns(2);sort=a.selectbox('Ordenar cuentas',['Cuenta','Budget','Divisa','Taxonomía']);desc=b.checkbox('Descendente')
  accounts=paged(repo.inventory(filters,sort,desc),'history')
  if accounts.empty:return
  account=st.selectbox('Serie de la página',accounts.account_id.tolist())
  history=data(repo.history(account));bands=st.checkbox('Mostrar bandas dinámicas ±1σ / ±2σ / ±3σ',True)
  st.plotly_chart(history_figure(history,bands),use_container_width=True)
  st.caption('Referencia acumulada de valores anteriores; mínimo cinco observaciones y sigma poblacional. Verde ±1σ, amarillo 1–2σ y rojo 2–3σ. Una referencia constante se muestra como no disponible.')
  st.download_button('Descargar historia y z-score de la cuenta',history.to_csv(index=False),file_name='historia_'+account+'.csv')
 elif page=='Model Arena':
  a,b,c=st.columns(3)
  options=list(targets)+(['Ambos'] if len(targets)>1 else [])
  target=a.selectbox('Objetivo',options,format_func=lambda name:targets.get(name,{}).get('label',name))
  metric=b.selectbox('Métrica',view['metrics'])
  fold=c.selectbox('Evaluación',['F1–F3','F1','F2','F3'])
  mode=st.radio('Resultado mostrado',['Mejor por input','Winner operativo'],horizontal=True)
  st.caption('F1–F3 participaron en selección. Los filtros consultan las asignaciones operativas publicadas.')
  raw=data(repo.metrics(filters,None if fold=='F1–F3' else fold))
  if raw.empty:st.info('No hay métricas evaluadas para esta población. Las asignaciones zero-shot no implican evaluación individual de todos los candidatos.');return
  if currency=='Todas':
   st.info('Cada divisa se mantiene en su propio panel.');panel=st.selectbox('Divisa del panel',sorted(raw.currency.unique()));raw=raw[raw.currency==panel]
  ranking=st.radio('Elegir la configuración por',['Universo filtrado','Cada budget'],horizontal=True) if mode=='Mejor por input' else 'Cada budget'
  presentation=st.radio('Vista de resultados',['Gráfico','Tabla'],horizontal=True,key='arena_presentation')
  objectives=list(targets) if target=='Ambos' else [target]
  panels=st.columns(len(objectives))
  for objective,panel in zip(objectives,panels):
   with panel:
    definition=targets[objective]
    st.subheader(definition['label'])
    if definition.get('unit'):st.caption(definition['unit'])
    table=comparisons(raw,metric,objective,mode,ranking)
    if table.empty:
     st.info('Sin comparación pareada disponible para este objetivo.')
     continue
    tabs=st.tabs(['Baseline','Modelo','Baseline vs Modelo'])
    for tab,column in zip(tabs,['baseline_value','value','improvement']):
     with tab:
      if presentation=='Gráfico':
       matrix=table.pivot_table(index='budget',columns='input',values=column,aggfunc='mean')
       color='RdBu' if column=='improvement' else 'Blues'
       fig=go.Figure(go.Heatmap(x=matrix.columns,y=matrix.index,z=matrix.to_numpy(),
                               colorscale=color,zmid=0 if column=='improvement' else None,
                               texttemplate='%{z:.3f}'))
       fig.update_layout(template='plotly_white',xaxis_title='Input',yaxis_title='Budget')
       st.plotly_chart(fig,use_container_width=True,key=f'arena_{objective}_{column}')
      else:
       columns=['budget','input','candidate','accounts','n',column]
       if column=='improvement':columns+=['baseline_value','value','difference','better']
       st.dataframe(table[columns],hide_index=True,use_container_width=True)
    st.download_button('Descargar comparación filtrada',table.to_csv(index=False),
                       file_name=f'baseline_winner_{objective}.csv',key=f'arena_download_{objective}')
  if mode=='Winner operativo' and st.checkbox('Explorar asignaciones operativas por cuenta'):
   accounts=paged(repo.inventory(filters),'routing')
   if not accounts.empty:
    account=st.selectbox('Cuenta para ver sus ganadores',accounts.account_id.tolist())
    row=accounts.loc[accounts.account_id==account].iloc[0]
    routes=pd.DataFrame(winner_rows(account,row.candidate,row.selection_source,objectives))
    st.dataframe(routes,hide_index=True,use_container_width=True)
    if st.checkbox('Ver pronósticos de esta cuenta'):
     daily=data(repo.daily(account))
     st.dataframe(daily[daily.target.isin(objectives)],hide_index=True,use_container_width=True)
  st.subheader('Modelo y entrenamiento')
  st.caption('Curvas reales por época: entrenamiento fijo y validación temporal. Total, budget y cuenta usan las evaluaciones registradas; F1–F3 participaron en selección, no son Test independiente.')
  scope=st.radio('Ámbito de las loss',['Total','Budget','Cuenta'],horizontal=True)
  if scope=='Budget':
   selected_budget=st.selectbox('Budget de las loss',list(range(1,11)))
   curve_filters=Filters(currency=filters.currency,budget=int(selected_budget))
   try:curves=data(repo.learning_entities(curve_filters))
   except Exception:curves=pd.DataFrame()
  elif scope=='Cuenta':
   selected_account=st.text_input('Cuenta de las loss')
   try:curves=data(repo.learning_entities(account=selected_account)) if selected_account else pd.DataFrame()
   except Exception:curves=pd.DataFrame()
  else:
   try:curves=data(repo.learning_entities(Filters(currency=filters.currency)))
   except Exception:
    try:curves=data(repo.learning_curves())
    except Exception:curves=pd.DataFrame()
  if curves.empty:st.info('Esta publicación no guardó entrenamiento fijo y validación por época. Disponible al ejecutar una nueva corrida instrumentada; no puede reconstruirse desde los pesos finales.')
  else:
   options=sorted(curves.candidate.str.split('_').str[0].unique())
   inp=st.radio('Input de las curvas',options,horizontal=True)
   curves=curves[curves.candidate.str.startswith(inp+'_')]
   fold=st.radio('Fold de las curvas',['F1','F2','F3','Todos','FUTURO'],horizontal=True)
   curves=curves[curves.fold.isin(['F1','F2','F3'])] if fold=='Todos' else curves[curves.fold==fold]
   names=sorted(curves.candidate.unique())
   chosen=st.multiselect('Modelos a comparar',names,default=names[:1])
   curves=curves[curves.candidate.isin(chosen)]
   display=st.radio('Vista de las curvas',['Gráfico','Tabla'],horizontal=True)
   if display=='Tabla':st.dataframe(curves,hide_index=True,use_container_width=True)
   else:
    from .classic_charts import figure
    st.plotly_chart(figure(curves),use_container_width=True)
   if fold=='FUTURO':st.caption('FUTURO usa toda la historia para entrenar: sólo hay curva de entrenamiento; no se inventa validación sin observado.')
  with st.expander('Configuraciones de todos los candidatos'):
   configs=data(repo.candidates());st.dataframe(configs,hide_index=True,use_container_width=True)
  with st.expander('Intentos HPO · completados y podados'):
   if st.button('Ver estados de todos los trials'):
    try:
     audit=data(repo.hpo_trials())
     if audit.empty:st.info('Esta publicación no incluye el registro de estados de HPO.')
     else:
      import plotly.express as px
      audit['encoder_kind']=audit.parameters_json.map(lambda value:json.loads(value).get('encoder_kind',json.loads(value).get('family','tft_compact')))
      counts=audit.groupby(['input','encoder_kind','state']).size().reset_index(name='trials')
      st.plotly_chart(px.bar(counts,x='input',y='trials',color='state',barmode='stack',labels={'input':'Input','trials':'Intentos','state':'Estado'}),use_container_width=True)
      st.dataframe(audit,hide_index=True,use_container_width=True)
      st.caption('PRUNED conserva las loss parciales y parámetros; sólo COMPLETE compite por ganador con F1–F3 completos. La última loss intermedia no es el score final.')
    except Exception as exc:st.warning('No se pudo leer el registro de HPO de esta publicación: '+str(exc))

 else:
  from .alerts_view import render_alerts
  render_alerts(repo,data,paged,filters)
