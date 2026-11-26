from dataclasses import dataclass
from .common import literal,shard

@dataclass(frozen=True)
class Filters:
 currency:str|None=None
 taxonomy:str|None=None
 budget:int|None=None
 cohort:str|None=None
 search:str=''

def predicates(filters,alias=''):
 p=alias+'.' if alias else ''
 terms=[]
 for name in ('currency','taxonomy','budget','cohort'):
  value=getattr(filters,name)
  if value is not None:terms.append(p+name+'='+literal(value))
 if filters.search:terms.append(f'strpos(lower({p}account_id),lower({literal(filters.search)})) > 0')
 return terms

class Repository:
 def __init__(self,config,run_id,publication_id,shards=None,*,mac2_events=False):
  self.config=config;self.run_id=run_id;self.publication_id=publication_id;self.shards=shards or config.shards;self.mac2_events=mac2_events
 def table(self,name):return self.config.table(name)
 def scope(self,alias=''):
  p=alias+'.' if alias else ''
  return [p+'run_id='+literal(self.run_id),p+'publication_id='+literal(self.publication_id)]
 def where(self,more=(),filters=None,alias=''):
  return ' AND '.join([*self.scope(alias),*(predicates(filters,alias) if filters else []),*more])
 def inventory(self,filters=Filters(),sort='Cuenta',descending=False):
  columns={'Cuenta':'account_id','Budget':'budget','Divisa':'currency','Taxonomía':'taxonomy'}
  if sort not in columns:raise ValueError('Orden no permitido')
  direction='DESC' if descending else 'ASC'
  return f'SELECT account_id,currency,taxonomy,budget,cohort,observations,candidate,selection_source FROM {self.table("inventory")} WHERE {self.where(filters=filters)} ORDER BY {columns[sort]} {direction} NULLS LAST,account_id ASC'
 def summary(self,filters=Filters()):
  return f'SELECT taxonomy,budget,count(*) AS accounts FROM {self.table("inventory")} WHERE {self.where(filters=filters)} GROUP BY taxonomy,budget ORDER BY taxonomy,budget'
 def history(self,account):
  return f'SELECT * FROM {self.table("history")} WHERE {self.where(["account_id="+literal(account),"segment="+literal(shard(account,self.shards))])} ORDER BY date,target'
 def daily(self,account):
  return f'SELECT * FROM {self.table("daily")} WHERE {self.where(["account_id="+literal(account),"segment="+literal(shard(account,self.shards))])} ORDER BY date,target,fold'
 def metrics(self,filters=Filters(),fold=None):
  keys='role,candidate,comparison_id,input,budget,target,currency'
  moments='n,mape_n,ae_sum,se_sum,smape_sum,mape_sum,covered_sum,width_sum,ae_scaled_sum,se_scaled_sum'.split(',')
  if filters.search:raise ValueError('La comparación agregada no admite búsqueda de cuenta; usa Historia o Alertamientos')
  more=['fold='+literal(fold or 'ALL')]
  # Metrics are already paired by account/date/fold at preparation time.
  return f'SELECT {keys},sum(accounts) AS accounts,'+','.join('sum('+m+') AS '+m for m in moments)+f' FROM {self.table("arena_summary")} WHERE {self.where(more,filters)} GROUP BY {keys} ORDER BY budget,input,candidate,target'
 def candidates(self):return f'SELECT * FROM {self.table("candidates")} WHERE {self.where()} ORDER BY input,candidate'
 def hpo_trials(self):return f'SELECT * FROM {self.table("hpo_trials")} WHERE {self.where()} ORDER BY input,trial_number'
 def training(self,candidate,fold=None):
  more=['candidate='+literal(candidate)]+(['fold='+literal(fold)] if fold else [])
  return f'SELECT candidate,fold,epoch,train_loss,validation_loss FROM {self.table("training")} WHERE {self.where(more)} ORDER BY fold,epoch'
 def events(self,filters=Filters(),source='Winner',fold=None,target=None,severity=None,cause=None,account=None,include_rules=False):
  source_filter = 'source IN ('+literal(source)+",'Regla contable')" if include_rules and source in ('Winner','Baseline') else 'source='+literal(source)
  more=[source_filter]
  if source=='Winner' and not self.mac2_events:
   winner = ("CASE WHEN substr(i.candidate,1,1)='{' THEN CASE d.target "
             "WHEN 'saldo' THEN try(json_extract_scalar(i.candidate,'$.saldo')) "
             "WHEN 'variacion' THEN try(json_extract_scalar(i.candidate,'$.variacion')) END "
             "ELSE i.candidate END")
   more.append("(source='Regla contable' OR (account_id,date,fold,target) IN ("
               "SELECT d.account_id,d.date,d.fold,d.target FROM "+self.table('daily')+
               " d JOIN "+self.table('inventory')+
               " i ON d.account_id=i.account_id AND d.run_id=i.run_id AND d.publication_id=i.publication_id WHERE "+
               self.where(alias='d')+" AND d.candidate="+winner+"))")
  for field,value in [('fold',fold),('account_id',account)]:
   if value:more.append(field+'='+literal(value))
  if target in ('saldo','variacion'):more.append('target='+literal(target))
  if severity:more.append('(severity='+literal(severity)+' OR overdraft=true)')
  if cause=='Sobregiro':more.append('overdraft=true')
  elif cause=='Alertamiento':more.append('has_alert=true')
  from .schemas import SCHEMAS
  alert='has_alert';overdraft='overdraft';reasons='reasons'
  if severity:
   alert='(has_alert AND severity='+literal(severity)+')'
   reasons='CASE WHEN severity='+literal(severity)+" THEN reasons ELSE 'ALERTA_SOBREGIRO' END"
  if cause=='Sobregiro':alert='false';reasons="'ALERTA_SOBREGIRO'"
  elif cause=='Alertamiento':
   overdraft='false';reasons="replace(reasons,'|ALERTA_SOBREGIRO','')"
   if severity:more.append('severity='+literal(severity))
  projection=','.join({'has_alert':alert+' AS has_alert','overdraft':overdraft+' AS overdraft','reasons':reasons+' AS reasons'}.get(k,k) for k in SCHEMAS['events'].names)
  sql=f'SELECT {projection} FROM {self.table("events")} WHERE {self.where(more,filters)}'
  if target=='Ambos el mismo día':
   sql='WITH filtered AS ('+sql+'), both_targets AS (SELECT account_id,date FROM filtered WHERE has_alert=true GROUP BY account_id,date HAVING count(DISTINCT target)=2) SELECT f.* FROM filtered f JOIN both_targets b ON f.account_id=b.account_id AND f.date=b.date'
  return sql
 def event_summary(self,**kwargs):
  return "SELECT count(DISTINCT CASE WHEN has_alert AND severity IN ('MUY BAJO','BAJO','MEDIO','ALTO') THEN account_id END) AS alert_accounts,count(DISTINCT CASE WHEN overdraft THEN account_id END) AS overdraft_accounts,count(*) AS events FROM ("+self.events(**kwargs)+') e'
 def event_accounts(self,**kwargs):
  return 'SELECT account_id,currency,budget,taxonomy,max(CASE WHEN has_alert THEN 1 ELSE 0 END) AS has_alert,max(CASE WHEN overdraft THEN 1 ELSE 0 END) AS overdraft FROM ('+self.events(**kwargs)+') e GROUP BY account_id,currency,budget,taxonomy ORDER BY account_id'
 def delivery(self,engine,fold=None):
  from .schemas import ENGINE2_COLUMNS
  from .mac_rules_reference import OFFICIAL_COLUMNS
  names={'engine2':ENGINE2_COLUMNS,'engine3':OFFICIAL_COLUMNS}
  if engine not in names:raise ValueError(engine)
  extra=['segment='+literal(fold)] if fold else []
  return 'SELECT '+','.join(names[engine])+f' FROM {self.table(engine)} WHERE {self.where(extra)} ORDER BY cutoff_date,cross_key_id'

 def severity_counts(self,**kwargs):
  rank="CASE WHEN has_alert THEN CASE severity WHEN 'MUY BAJO' THEN 1 WHEN 'BAJO' THEN 2 WHEN 'MEDIO' THEN 3 WHEN 'ALTO' THEN 4 ELSE 0 END ELSE 0 END"
  return 'SELECT severity_rank,count(*) AS accounts FROM (SELECT account_id,max('+rank+') AS severity_rank FROM ('+self.events(**kwargs)+') e GROUP BY account_id) g WHERE severity_rank>0 GROUP BY severity_rank ORDER BY severity_rank'

 def training_targets(self,fold=None):
  more=['fold='+literal(fold)] if fold else []
  return 'SELECT candidate,fold,target,epoch,train_loss FROM '+self.table('training_targets')+' WHERE '+self.where(more)+' ORDER BY candidate,fold,target,epoch'

 def learning_curves(self):
  return 'SELECT candidate,fold,target,epoch,split,loss FROM '+self.table('training_learning')+' WHERE '+self.where()+' ORDER BY candidate,fold,target,split,epoch'

 def learning_entities(self,filters=Filters(),account=None):
  more=['account_id='+literal(account)] if account else []
  keys='candidate,fold,target,epoch,split'
  return ('SELECT '+keys+',sum(loss*n_observations)/sum(n_observations) AS loss,count(DISTINCT account_id) AS accounts FROM '+self.table('training_entities')+' WHERE '+self.where(more,filters)+' GROUP BY '+keys+' ORDER BY '+keys)
