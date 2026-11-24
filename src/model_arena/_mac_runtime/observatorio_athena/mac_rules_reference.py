"""Literal portable contract of MAC Engine 3; validated against its Spark source.

No future-risk extensions, adjusted severities, or model selection live here.
"""

import math
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation

import polars as pl

PROTOCOL = "mac-engine3-literal-v1"
REFERENCE_SHA256 = "d01afaef61a59f066229b72f3a19429b407901b573f3e744455eae2f0dca54ed"
DECIMAL = pl.Decimal(24, 8)
DECIMAL_FIELDS = {
    "gl_mthly_ab_mxn_amount": "saldo_del_dia",
    "previous_day_balance_amount": "saldo_anterior",
    "prev_day_bal_cbl_vary_amount": "variacion_saldo",
    "gl_mthly_ab_mxn1_amount": "saldo_del_dia_valorizado",
    "previous_day_balance1_amount": "saldo_del_dia_anterior_valorizado",
    "prev_day_bal_cbl_vary1_amount": "variacion_valorizada",
    "fix_exchange_amount": "tipo_cambio_final",
    "dmlm_ranges_curr_balance_per": "porcentaje_alerta",
}
OFFICIAL_COLUMNS = (
    "cutoff_date",
    "gl_account_id",
    "currency_id",
    "cross_key_id",
    "gl_mthly_ab_mxn_amount",
    "previous_day_balance_amount",
    "prev_day_bal_cbl_vary_amount",
    "classification_account_type",
    "gl_account_hist_behaviour_type",
    "gl_mthly_ab_mxn1_amount",
    "previous_day_balance1_amount",
    "prev_day_bal_cbl_vary1_amount",
    "fix_exchange_amount",
    "dmlm_rng_excd_bal_alert_type",
    "dmlm_ranges_curr_balance_per",
    "alert_status_type",
    "overdraft_type",
    "data_mach_learning_model_type",
    "load_date",
    "dmlm_business_attributes_desc",
    "dmlm_business_attributes1_desc",
    "dmlm_bus_atrb_value_number",
)


def spark_decimal(value):
    """Spark non-ANSI cast to Decimal(24,8), including overflow -> null."""
    if value is None or value == "" or not math.isfinite(float(value)):
        return None
    try:
        result = Decimal(str(value)).quantize(
            Decimal("0.00000001"), rounding=ROUND_HALF_UP
        )
    except InvalidOperation:
        return None
    return result if abs(result) < Decimal(10000000000000000) else None


def _severity(percent, *, atypical):
    out = pl.when((percent > 0) & (percent < 0.25)).then(
        pl.lit("BAJO" if atypical else "MUY BAJO")
    )
    if atypical:
        return (
            out.when((percent >= 0.25) & (percent < 0.50))
            .then(pl.lit("MEDIO"))
            .when(percent >= 0.50)
            .then(pl.lit("ALTO"))
            .otherwise(pl.lit("OK"))
        )
    return (
        out.when((percent >= 0.25) & (percent < 0.50))
        .then(pl.lit("BAJO"))
        .when((percent >= 0.50) & (percent < 0.80))
        .then(pl.lit("MEDIO"))
        .when(percent >= 0.80)
        .then(pl.lit("ALTO"))
        .otherwise(pl.lit("OK"))
    )


@dataclass(frozen=True)
class MacEvaluation:
    training_balance: pl.DataFrame
    training_variation: pl.DataFrame
    atypical_balance: pl.DataFrame
    atypical_variation: pl.DataFrame
    zero: pl.DataFrame
    official: pl.DataFrame


class OriginalMacRules:
    """One original job day, or explicitly partitioned repetitions of that job."""

    def run(self, validation):
        if validation["cutoff_date"].n_unique() != 1:
            raise ValueError("original MAC run requires exactly one cutoff date")
        return self.run_partitioned(validation, context=("cutoff_date",))

    def run_partitioned(self, validation, *, context, atypical_as_trainable=False):
        if "cutoff_date" not in context or len(set(context)) != len(context):
            raise ValueError("partition keys must uniquely include cutoff_date")
        classes = pl.col("gl_account_hist_behaviour_type")
        train = validation.filter(classes.is_in(["cuentas_entrenamiento", "cuentas_atipicos"]) if atypical_as_trainable else classes == "cuentas_entrenamiento")
        atypical = validation.filter(pl.lit(False) if atypical_as_trainable else classes == "cuentas_atipicos")
        zeros = validation.filter(classes == "cuentas_cero")
        balance = self._bounds(
            train.filter(
                ~pl.col("cross_key_id").str.ends_with("_var").fill_null(False)
            ),
            variation=False,
            atypical=False,
        )
        variation = self._bounds(
            train.filter(pl.col("cross_key_id").str.ends_with("_var")),
            variation=True,
            atypical=False,
        )
        atyp_balance = self._bounds(atypical, variation=False, atypical=True)
        atyp_variation = self._bounds(atypical, variation=True, atypical=True)
        amount = pl.col("saldo_del_dia_valorizado").abs()
        zeros = zeros.with_columns(
            pl.when((amount > 500000) & (amount < 1000000))
            .then(pl.lit("BAJO"))
            .when((amount >= 1000000) & (amount < 5000000))
            .then(pl.lit("MEDIO"))
            .when(amount >= 5000000)
            .then(pl.lit("ALTO"))
            .otherwise(pl.lit("OK"))
            .alias("niveles_alerta")
        )
        keys = list(dict.fromkeys([*context, "gl_account_id", "currency_id"]))
        parts = []
        for saldo, var in ((balance, variation), (atyp_balance, atyp_variation)):
            # The original selects only identity from variation before joining;
            # all values in the official record come from the balance row.
            confirmed = (
                var.filter(pl.col("niveles_alerta") != "OK")
                .select(keys)
                .join(
                    saldo.filter(pl.col("niveles_alerta") != "OK"), on=keys, how="inner"
                )
            )
            parts.extend([confirmed, saldo.filter(pl.col("alerta_sobregiro") != "OK")])
        parts.append(
            zeros.filter(pl.col("niveles_alerta") != "OK").with_columns(
                pl.lit("").alias("alertamiento"),
                pl.lit(None, pl.Float64).alias("porcentaje_alerta"),
                pl.lit("").alias("alerta_sobregiro"),
            )
        )
        records = pl.concat(parts, how="diagonal_relaxed")
        official = self._output(records, context)
        return MacEvaluation(
            balance, variation, atyp_balance, atyp_variation, zeros, official
        )

    @staticmethod
    def _bounds(frame, *, variation, atypical):
        observed = pl.col("variacion_saldo" if variation else "saldo_del_dia")
        if atypical:
            spread = pl.col(
                "stddev_variacion_saldo" if variation else "stddev_saldo_del_dia"
            )
            lower, upper = observed - 1.959964 * spread, observed + 1.959964 * spread
        else:
            lower, upper = pl.col("lower_interval"), pl.col("upper_interval")
        if not variation:
            lower = (
                pl.when(
                    (pl.col("classification_account_type") == "positiva") & (lower < 0)
                )
                .then(0.0)
                .otherwise(lower)
            )
            upper = (
                pl.when(
                    (pl.col("classification_account_type") == "negativa") & (upper > 0)
                )
                .then(0.0)
                .otherwise(upper)
            )
        frame = frame.with_columns(
            lower.alias("adjust_lower_interval"), upper.alias("adjust_upper_interval")
        )
        frame = frame.with_columns(
            pl.when(observed < pl.col("adjust_lower_interval"))
            .then(pl.lit("ALERTA_INFERIOR"))
            .when(observed > pl.col("adjust_upper_interval"))
            .then(pl.lit("ALERTA_SUPERIOR"))
            .otherwise(pl.lit("OK"))
            .alias("alertamiento")
        )
        boundary = (
            pl.when(pl.col("alertamiento") == "ALERTA_INFERIOR")
            .then(pl.col("adjust_lower_interval"))
            .otherwise(pl.col("adjust_upper_interval"))
        )
        percentage = (
            pl.when(pl.col("alertamiento") == "OK")
            .then(0.0)
            .when(boundary == 0)
            .then(None)
            .otherwise((observed / boundary - 1).abs())
        )
        frame = frame.with_columns(percentage.alias("porcentaje_alerta"))
        frame = frame.with_columns(
            _severity(pl.col("porcentaje_alerta"), atypical=atypical).alias(
                "niveles_alerta"
            )
        )
        if not variation:
            frame = frame.with_columns(
                pl.when(
                    (
                        (observed < 0)
                        & (pl.col("classification_account_type") == "positiva")
                    )
                    | (
                        (observed > 0)
                        & (pl.col("classification_account_type") == "negativa")
                    )
                )
                .then(pl.lit("ALERTA_SOBREGIRO"))
                .otherwise(pl.lit("OK"))
                .alias("alerta_sobregiro")
            )
        return frame

    @staticmethod
    def _output(records, context):
        fields = [
            pl.col("cutoff_date"),
            pl.col("gl_account_id"),
            pl.col("currency_id"),
            pl.concat_str("gl_account_id", "currency_id").alias("cross_key_id"),
            pl.col("classification_account_type"),
            pl.col("gl_account_hist_behaviour_type"),
            pl.col("alertamiento").alias("dmlm_rng_excd_bal_alert_type"),
            pl.col("niveles_alerta").alias("alert_status_type"),
            pl.col("alerta_sobregiro").alias("overdraft_type"),
            pl.col("final_model").alias("data_mach_learning_model_type"),
            pl.col("cutoff_date").alias("load_date"),
            pl.lit("").alias("dmlm_business_attributes_desc"),
            pl.lit("").alias("dmlm_business_attributes1_desc"),
            pl.lit(None, DECIMAL).alias("dmlm_bus_atrb_value_number"),
        ]
        extras = [key for key in context if key not in OFFICIAL_COLUMNS]
        result = records.select(*fields, *[pl.col(key) for key in extras])
        for target, source in DECIMAL_FIELDS.items():
            result = result.with_columns(
                pl.Series(
                    target, [spark_decimal(v) for v in records[source]], dtype=DECIMAL
                )
            )
        return result.select(*OFFICIAL_COLUMNS, *extras).unique(maintain_order=True)
