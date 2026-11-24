"""Budget finalists and chronological series assignment using seen evidence."""
import polars as pl


def assignments(scores, inventory, candidates, prior_folds):
    eligible = scores.filter(pl.col('fold').is_in(prior_folds))
    if eligible.is_empty():
        return inventory.with_columns(pl.lit(candidates[0].id).alias('candidate'),pl.lit('INITIAL_CONFIG').alias('selection_source'))
    finalists = (eligible.group_by('budget','input','candidate').agg(pl.col('score').mean())
        .sort('budget','input','score','candidate').unique(['budget','input'],keep='first'))
    global_finalists = (eligible.group_by('input','candidate').agg(pl.col('score').mean())
        .sort('input','score','candidate').unique('input',keep='first'))
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
