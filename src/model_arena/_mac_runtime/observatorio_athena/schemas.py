import pyarrow as pa
from .mac_rules_reference import OFFICIAL_COLUMNS, DECIMAL_FIELDS

ENGINE2_COLUMNS = ('load_date','cutoff_date','cross_key_id','data_mach_learning_model_type',
 'dmlm_gl_account_pntb_amount','dmlm_lower_interval_number','dmlm_upper_interval_number',
 'dmlm_mean_absolute_error_per','dmlm_symc_mean_abs_error_per','dmlm_mean_abs_error_number',
 'dmlm_mean_square_error_number','dmlm_root_mean_sq_error_number','dmlm_business_attributes_desc',
 'dmlm_business_attributes1_desc','dmlm_bus_atrb_value_number')
S,D,I,B,T = pa.string(),pa.float64(),pa.int64(),pa.bool_(),pa.date32()
def schema(fields):return pa.schema(fields)
SCHEMAS={
 'inventory':schema([(k,S) for k in ['account_id','currency','taxonomy','cohort','sign','candidate','selection_source']]+[('budget',I),('observations',I)]),
 'history':schema([(k,S) for k in ['account_id','currency','target','zscore_status']]+[('date',T)]+[(k,D) for k in ['observed','sigma_center','sigma_std','zscore','hist_q025','hist_q50','hist_q975']]+[('sigma_reference_n',I)]),
 'daily':schema([(k,S) for k in ['account_id','currency','target','fold','candidate','input','taxonomy','sign']]+[('budget',I),('date',T),('origin',T),('fx',D)]+[(k,D) for k in ['observed','model_q025','model_q50','model_q975','baseline_q025','baseline_q50','baseline_q975']]),
 'events':schema([(k,S) for k in ['account_id','currency','taxonomy','cohort','target','fold','source','severity','reasons']]+[('budget',I),('date',T),('has_alert',B),('overdraft',B)]+[(k,D) for k in ['observed','lower','upper','percentage']]),
 'metrics':schema([(k,S) for k in ['account_id','currency','cohort','taxonomy','target','fold','input','comparison_id','candidate','role','population_hash']]+[('budget',I),('n',I),('mape_n',I)]+[(k,D) for k in ['ae_sum','se_sum','smape_sum','mape_sum','covered_sum','width_sum','ae_scaled_sum','se_scaled_sum']]),
 'candidates':schema([(k,S) for k in ['candidate','input','parameters_json']]+[('selection_score',D)]),
 'training':schema([(k,S) for k in ['candidate','fold']]+[('epoch',I),('train_loss',D),('validation_loss',D)]),
 'publications':schema([(k,S) for k in ['run_id','publication_id','published_at','source_signature','manifest_uri','protocol']]+[('shards',I)]),
}
SCHEMAS['engine2']=schema([(k,T if k in ('load_date','cutoff_date') else pa.decimal128(10,8) if k in ('dmlm_mean_absolute_error_per','dmlm_symc_mean_abs_error_per') else pa.decimal128(24,8) if k.startswith('dmlm_') and k not in ('dmlm_business_attributes_desc','dmlm_business_attributes1_desc') else S) for k in ENGINE2_COLUMNS])
SCHEMAS['engine3']=schema([(k,T if k in ('cutoff_date','load_date') else pa.decimal128(24,8) if k in (*DECIMAL_FIELDS,'dmlm_bus_atrb_value_number') else S) for k in OFFICIAL_COLUMNS])


def athena_type(t):
 if pa.types.is_string(t):return 'string'
 if pa.types.is_float64(t):return 'double'
 if pa.types.is_int64(t):return 'bigint'
 if pa.types.is_boolean(t):return 'boolean'
 if pa.types.is_date(t):return 'date'
 if pa.types.is_decimal(t):return f'decimal({t.precision},{t.scale})'
 raise TypeError(str(t))

SCHEMAS['arena_summary']=schema([(k,S) for k in ['currency','cohort','taxonomy','target','fold','input','comparison_id','candidate','role']]+[('budget',I),('accounts',I),('n',I),('mape_n',I)]+[(k,D) for k in ['ae_sum','se_sum','smape_sum','mape_sum','covered_sum','width_sum','ae_scaled_sum','se_scaled_sum']])

# Separate dataset: existing obs_training schema stays compatible.
SCHEMAS['training_targets']=schema([('candidate',S),('fold',S),('target',S),('epoch',I),('train_loss',D)])

SCHEMAS['training_learning']=schema([('candidate',S),('fold',S),('target',S),('epoch',I),('split',S),('loss',D)])

SCHEMAS['training_entities']=schema([(k,S) for k in ['account_id','currency','cohort','taxonomy','candidate','fold','target','split']]+[('budget',I),('epoch',I),('loss',D),('n_observations',I)])

# Additive provenance dataset; existing publications and MAC schemas are unchanged.
SCHEMAS['calendar_provenance']=schema([(k,S) for k in ['generator','generator_version','profile','view','start_date','end_date','calendar_sha256','metadata_json']]+[('feature_count',I),('rows',I)])

SCHEMAS['hpo_trials']=schema([(k,S) for k in ['input','candidate','state','pruned_fold','parameters_json']]+[('trial_number',I),('pruned_epoch',I),('objective_value',D),('last_intermediate_value',D)])
