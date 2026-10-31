from __future__ import annotations

import re
from collections import defaultdict
from typing import Mapping, Sequence


_PUMP_PATTERN = re.compile(r"(?:^|_)PU(\d+)$")


def pump_component_from_tag(tag: str) -> str | None:
    """Map BATADAL pump telemetry tags such as F_PU10 to their pump component."""
    match = _PUMP_PATTERN.search(str(tag))
    return f"PU{int(match.group(1))}" if match else None


def aggregate_component_scores(tag_scores: Mapping[str, float]) -> dict[str, float]:
    """Aggregate nonnegative feature scores to physical pump components."""
    scores: dict[str, float] = defaultdict(float)
    for tag, value in tag_scores.items():
        component = pump_component_from_tag(tag)
        if component is not None:
            scores[component] += max(0.0, float(value))
    return dict(scores)


def rank_components(component_scores: Mapping[str, float]) -> list[str]:
    """Rank components by descending aggregated contribution, then stable name."""
    return [name for name, _ in sorted(component_scores.items(), key=lambda item: (-item[1], item[0]))]


def component_hit_at_k(ranking: Sequence[str], targets: Sequence[str], k: int) -> bool:
    if k < 1:
        raise ValueError("k must be positive")
    return bool(set(ranking[:k]) & set(targets))
