"""Classification version events; these are not MAC alerts or behavior change dates."""

from collections import defaultdict


def classification_events(snapshots):
    groups = defaultdict(lambda: defaultdict(set))
    for row in snapshots:
        groups[row["cross_key_id"]][row["load_date"]].add(
            (row["currency_id"], row["taxonomy"], row["account_sign"])
        )
    events, ambiguous = [], []
    for account, loads in sorted(groups.items()):
        previous = None
        for load, labels in sorted(loads.items()):
            if len(labels) != 1:
                ambiguous.append(
                    {"account_id": account, "load_date": load, "variants": len(labels)}
                )
                previous = None
                continue
            label = next(iter(labels))
            if previous and label != previous[1]:
                events.append(
                    {
                        "account_id": account,
                        "published_load_date": load,
                        "previous_load_date": previous[0],
                        "before": list(previous[1]),
                        "after": list(label),
                        "type": "LOAD_SNAPSHOT_RECLASSIFICATION",
                        "mac_alert": False,
                        "behavior_change_date": None,
                    }
                )
            previous = (load, label)
    return {
        "status": "AVAILABLE",
        "scope": "Selected accounts and loads overlapping the historical window",
        "date_semantics": "Publication load date; not the inferred onset of a change in balance behavior",
        "label_order": ["currency_id", "taxonomy", "account_sign"],
        "snapshots": snapshots,
        "events": events,
        "ambiguous_snapshots": ambiguous,
    }
