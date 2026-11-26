"""Expose the existing Engine 3 decision in the account event tables."""

import polars as pl

from .mac_rules_reference import confirmed_variation_keys


def mac2_events(evaluation, *, source, fold, accounts, include_zeros=False):
    keys = ['cutoff_date', 'gl_account_id', 'currency_id']
    rows = []
    for balance, variation in (
        (evaluation.training_balance, evaluation.training_variation),
        (evaluation.atypical_balance, evaluation.atypical_variation),
    ):
        # Use the same confirmation used by Engine 3 and preserve both
        # contributing diagnostics for the monitor's objective filter.
        confirmed = (
            balance.filter(pl.col('niveles_alerta') != 'OK')
            .select(*keys, pl.col('niveles_alerta').alias('mac2_severity'))
            .join(confirmed_variation_keys(variation, keys),
                  on=keys, how='inner').unique()
        )
        selected = (
            (balance.join(confirmed,
                          on=keys, how='left')
             .filter(pl.col('mac2_severity').is_not_null() | (pl.col('alerta_sobregiro') != 'OK')), 'saldo'),
            (variation.join(confirmed, on=keys, how='inner'), 'variacion'),
        )
        for frame, target in selected:
            for record in frame.iter_rows(named=True):
                aid = record['account_id']
                account = accounts[aid]
                has_alert = record['mac2_severity'] is not None
                overdraft = target == 'saldo' and record['alerta_sobregiro'] == 'ALERTA_SOBREGIRO'
                reasons = [record['alertamiento']] if has_alert else []
                if overdraft:
                    reasons.append('ALERTA_SOBREGIRO')
                rows.append(dict(
                    account_id=aid, currency=account['currency'], taxonomy=account['taxonomy'],
                    cohort=account['cohort'], budget=account['budget'],
                    date=record['cutoff_date'], target=target, fold=fold, source=source,
                    severity=record['mac2_severity'] if has_alert else 'OK',
                    reasons='|'.join(reasons), has_alert=has_alert, overdraft=overdraft,
                    observed=record['saldo_del_dia' if target == 'saldo' else 'variacion_saldo'],
                    lower=record['adjust_lower_interval'], upper=record['adjust_upper_interval'],
                    percentage=record['porcentaje_alerta'] if has_alert else None,
                ))
    if include_zeros:
        for record in evaluation.zero.filter(pl.col('niveles_alerta') != 'OK').iter_rows(named=True):
            aid = record['account_id']
            account = accounts[aid]
            rows.append(dict(
                account_id=aid, currency=account['currency'], taxonomy=account['taxonomy'],
                cohort=account['cohort'], budget=account['budget'],
                date=record['cutoff_date'], target='saldo', fold=fold, source='Regla contable',
                severity=record['niveles_alerta'], reasons='REGLA_CERO', has_alert=True,
                overdraft=False, observed=record['saldo_del_dia'], lower=None,
                upper=None, percentage=None,
            ))
    return rows
