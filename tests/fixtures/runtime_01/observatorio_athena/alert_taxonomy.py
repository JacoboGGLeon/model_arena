"""Diagnóstico por objetivo. Contrato separado de la salida MAC oficial."""
import math

SEVERITY_COLORS={'MUY BAJO':'#5b9bd5','BAJO':'#1464a5','MEDIO':'#e6b800','ALTO':'#c52c48','SIN_GRADUAR':'#607d8b','OK':'#aab7c4'}
SEVERITY_ORDER={'OK':0,'SIN_GRADUAR':1,'MUY BAJO':2,'BAJO':3,'MEDIO':4,'ALTO':5}
CONDITION_SYMBOLS={'ALERTA_INFERIOR':'▼','ALERTA_SUPERIOR':'▲','ALERTA_SOBREGIRO':'◆','REGLA_CERO':'■'}


def maximum_severity(records):
    """Display aggregation only: graded alerts first; no new MAC evaluation."""
    return max((r['severity'] for r in records),key=lambda s:SEVERITY_ORDER[s],default='OK')


def account_severity_counts(records):
    """One account, one bar: maximum severity among the supplied filtered rows."""
    from collections import defaultdict,Counter
    grouped=defaultdict(list)
    for record in records:grouped[record['cross_key_id']].append(record)
    return Counter(maximum_severity(rows) for rows in grouped.values())


def evaluate(value, lower, upper, sign, target, taxonomy, fx=1.):
    """Mismas desigualdades/severidades del port literal; atípicas usan cuantiles como E2E."""
    if target not in ('saldo','variacion'):raise ValueError(target)
    reasons=[];percent=None;severity='OK';overdraft=False
    if taxonomy=='cuentas_cero':
        if target=='variacion':return dict(active=False,reasons=[],severity='OK',lower=None,upper=None,percentage=None,overdraft=False)
        amount=abs(value*fx)
        severity='ALTO' if amount>=5000000 else 'MEDIO' if amount>=1000000 else 'BAJO' if amount>500000 else 'OK'
        if severity!='OK':reasons=['REGLA_CERO']
        return dict(active=bool(reasons),reasons=reasons,severity=severity,lower=None,upper=None,percentage=None,overdraft=False)
    if not all(v is not None and math.isfinite(float(v)) for v in (lower,upper,value)):raise ValueError('Faltan límites predictivos finitos')
    if lower>upper:raise ValueError('Límites invertidos')
    if target=='saldo':
        if sign=='positiva':lower=max(0.,lower)
        if sign=='negativa':upper=min(0.,upper)
        overdraft=(sign=='positiva' and value<0) or (sign=='negativa' and value>0)
    direction='ALERTA_INFERIOR' if value<lower else 'ALERTA_SUPERIOR' if value>upper else None
    if direction:
        reasons.append(direction); boundary=lower if direction=='ALERTA_INFERIOR' else upper
        if boundary==0:severity='SIN_GRADUAR'
        else:
            percent=abs(value/boundary-1)
            severity='ALTO' if percent>=.8 else 'MEDIO' if percent>=.5 else 'BAJO' if percent>=.25 else 'MUY BAJO' if percent>0 else 'OK'
    if overdraft:reasons.append('ALERTA_SOBREGIRO')
    return dict(active=bool(reasons),reasons=reasons,severity=severity,lower=lower,upper=upper,percentage=percent,overdraft=overdraft)


def event_label(r):
    conditions=' + '.join(CONDITION_SYMBOLS.get(x,'')+' '+x.replace('ALERTA_','').replace('REGLA_','') for x in r['reasons'])
    return f"{r['target']} · {conditions} · severidad {r['severity']}"


def scope(records):
    targets={r['target'] for r in records if r['active']}
    return 'Saldo y variación' if len(targets)==2 else 'Sólo saldo' if targets=={'saldo'} else 'Sólo variación' if targets=={'variacion'} else 'Sin alerta'
