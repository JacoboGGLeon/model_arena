"""Budget finalists and chronological series assignment using seen evidence."""
import polars as pl


def assignments(scores, inventory, candidates, prior_folds):
    if candidates and getattr(candidates[0], 'target', None):
        parts=[]
        for target in ('saldo','variacion'):
            pool=[c for c in candidates if c.target==target]
            if not pool:
                raise ValueError('Missing candidates for '+target)
            # Recursive legacy selector on exactly one objective, preserving fallback policy.
            from types import SimpleNamespace
            plain=[SimpleNamespace(**{k:v for k,v in vars(c).items() if k!='target'}) for c in pool]
            part=assignments(scores.filter(pl.col('target')==target),inventory,plain,prior_folds)
            parts.append(part.with_columns(pl.lit(target).alias('target')))
        return pl.concat(parts)
    eligible = scores.filter(pl.col('fold').is_in(prior_folds))
    if eligible.is_empty():
        return inventory.with_columns(pl.lit(candidates[0].id).alias('candidate'),pl.lit('INITIAL_CONFIG').alias('selection_source'))
    dimensions = ['family','input'] if 'family' in eligible.columns and eligible['family'].n_unique()>1 else ['input']
    finalists = (eligible.group_by('budget',*dimensions,'candidate').agg(pl.col('score').mean())
        .sort('budget',*dimensions,'score','candidate').unique(['budget',*dimensions],keep='first'))
    global_finalists = (eligible.group_by(*dimensions,'candidate').agg(pl.col('score').mean())
        .sort(*dimensions,'score','candidate').unique(dimensions,keep='first'))
    result = []
    for row in inventory.to_dicts():
        ids = finalists.filter(pl.col('budget')==row['budget'])['candidate'].to_list()
        if not ids:
            ids=global_finalists['candidate'].to_list()
        pool=eligible.filter(pl.col('candidate').is_in(ids))
        own=pool.filter(pl.col('account_id')==row['account_id']) if row['cohort']=='seen' else pl.DataFrame()
        if own.height:
            segment, source = own, 'SERIE'
        else:
            segment=pool.filter((pl.col('currency')==row['currency']) & (pl.col('budget')==row['budget']))
            source='DIVISA_BUDGET'
            if segment.is_empty():
                segment=pool.filter(pl.col('budget')==row['budget'])
                source='BUDGET'
            if segment.is_empty():
                segment=pool
                source='GLOBAL'
        choice=segment.group_by('candidate').agg(pl.col('score').mean()).sort('score','candidate')['candidate'][0]
        result.append(dict(**row,candidate=choice,selection_source=source))
    return pl.DataFrame(result)
