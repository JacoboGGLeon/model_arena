"""Quantile scoring and deterministic candidate ranking across profiles."""


def pinball_error(error, quantile, *, maximum):
    return maximum(quantile * error, (quantile - 1) * error)


def best_candidate(rows, *, score):
    return min(rows, key=lambda row: (row[score], row["candidate"]))
