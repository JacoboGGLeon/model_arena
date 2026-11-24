import pytest
from pydantic import ValidationError
from model_arena import ArenaSpec, ModelSpec, PanelSchema, TemporalCV
from model_arena.contracts import SearchSpec, TrainingSpec


@pytest.mark.parametrize("field,value", [("h", 0), ("window", 0), ("schema_version", "2"),
                                         ("quantiles", [.5, .025]), ("unexpected", 1)])
def test_invalid_arena_parameters_fail_before_work(field, value):
    with pytest.raises(ValidationError):
        ArenaSpec.model_validate({"models": [{"alias": "test"}], "freq": "D", field: value})


@pytest.mark.parametrize("value", [[], ["same", "same"]])
def test_target_contract(value):
    with pytest.raises(ValidationError):
        PanelSchema(target_cols=value)


@pytest.mark.parametrize("value", [(0, 1), (1, 0), (2, 1)])
def test_invalid_search_ranges(value):
    with pytest.raises(ValidationError):
        SearchSpec(lr_range=value)


def test_temporal_folds_cannot_silently_reuse_weights():
    with pytest.raises(ValidationError):
        TemporalCV(refit=False)


def test_explicit_calendar_required():
    with pytest.raises(ValidationError):
        ArenaSpec(models=[ModelSpec(alias="test")])


def test_training_config_rejects_typo():
    with pytest.raises(ValidationError):
        TrainingSpec(epocs=75)

