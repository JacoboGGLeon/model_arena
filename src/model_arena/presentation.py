"""Presentation metadata and winner routing for published artifacts."""

import json


def publication_view(targets, quantiles, *, labels=None, units=None, metrics=(), routing=()):
    return {
        "targets": [
            {"id": name, "label": (labels or {}).get(name, name), "unit": (units or {}).get(name)}
            for name in targets
        ],
        "quantiles": list(quantiles),
        "metrics": list(metrics),
        "routing": list(routing),
    }


def winner_rows(account, candidate, source, targets):
    """Decode 0.1.5 objective maps and older scalar winner assignments."""
    def mapping(value):
        if isinstance(value, str) and value.startswith("{"):
            try:
                decoded = json.loads(value)
            except ValueError:
                pass
            else:
                if isinstance(decoded, dict):
                    return decoded
        return dict.fromkeys(targets, value)

    candidates, sources = mapping(candidate), mapping(source)
    return [
        {"account_id": account, "target": target, "candidate": candidates.get(target),
         "selection_source": sources.get(target)}
        for target in targets
    ]
