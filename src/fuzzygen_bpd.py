"""Paper-based fuzzy Behavioral Patch Distance scoring utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd


FEATURES = ("TestDelta", "CoverageDelta", "DiffSize")
SINGLETONS = (0.1, 0.5, 0.9)
BREAKPOINTS = {
    "TestDelta": ((0.0, 0.0, 0.15, 0.30), (0.15, 0.30, 0.50, 0.65), (0.50, 0.65, 1.0, 1.0)),
    "CoverageDelta": ((0.0, 0.0, 0.02, 0.05), (0.02, 0.05, 0.10, 0.15), (0.10, 0.15, 1.0, 1.0)),
    "DiffSize": ((0.0, 0.0, 0.05, 0.15), (0.05, 0.15, 0.30, 0.45), (0.30, 0.45, 1.0, 1.0)),
}
GATES = ("build_pass", "pov_pass", "functional_pass")


def trapezoid(x: float, a: float, b: float, c: float, d: float) -> float:
    """Trapezoidal membership with the paper's shoulder conventions."""
    rising = 1.0 if a == b and x >= b else (0.0 if a == b else (x - a) / (b - a))
    falling = 1.0 if c == d and x <= c else (0.0 if c == d else (d - x) / (d - c))
    return float(max(0.0, min(rising, 1.0, falling)))


def fuzzify(feature: str, value: float) -> float:
    memberships = [trapezoid(value, *points) for points in BREAKPOINTS[feature]]
    total = sum(memberships)
    if total == 0:
        raise ValueError(f"Membership functions do not cover {feature}={value}.")
    return float(sum(mu * singleton for mu, singleton in zip(memberships, SINGLETONS)) / total)


def normalized_weights(raw_weights: tuple[float, float, float]) -> tuple[float, float, float]:
    if len(raw_weights) != len(FEATURES):
        raise ValueError(f"Expected exactly {len(FEATURES)} feature weights.")
    if not all(np.isfinite(weight) for weight in raw_weights):
        raise ValueError("Feature weights must be finite numbers.")
    if any(weight < 0 for weight in raw_weights):
        raise ValueError("Feature weights cannot be negative.")
    total = sum(raw_weights)
    if total <= 0:
        raise ValueError("At least one feature weight must be greater than zero.")
    return tuple(float(value / total) for value in raw_weights)


def score_candidates(data: pd.DataFrame, weights: tuple[float, float, float], threshold: float) -> pd.DataFrame:
    """Gate candidates, score eligible rows, and rank within each vulnerability."""
    result = data.copy()
    w = normalized_weights(weights)
    for feature in FEATURES:
        result[f"F_{feature}"] = result[feature].map(lambda value: fuzzify(feature, float(value)))
    result["eligible"] = result[list(GATES)].astype(bool).all(axis=1)
    result["BPD"] = np.nan
    eligible = result["eligible"]
    result.loc[eligible, "BPD"] = sum(
        w[i] * result.loc[eligible, f"F_{feature}"] for i, feature in enumerate(FEATURES)
    )
    result["decision"] = np.where(~eligible, "Ineligible", np.where(result["BPD"] < threshold, "Lower disturbance", "Higher disturbance"))
    result["rank"] = pd.Series(pd.NA, index=result.index, dtype="Int64")
    result.loc[eligible, "rank"] = (
        result.loc[eligible].groupby("vulnerability_id")["BPD"].rank(method="min", ascending=True).astype("Int64")
    )
    return result.sort_values(["vulnerability_id", "eligible", "BPD"], ascending=[True, False, True], na_position="last")


def validate_candidates(data: pd.DataFrame) -> pd.DataFrame:
    required = {"vulnerability_id", "candidate_id", *FEATURES, *GATES}
    missing = required - set(data.columns)
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(sorted(missing)))
    result = data.copy()
    for column in FEATURES:
        result[column] = pd.to_numeric(result[column], errors="coerce")
        if result[column].isna().any() or ((result[column] < 0) | (result[column] > 1)).any():
            raise ValueError(f"{column} values must be numeric and between 0 and 1.")
    for column in GATES:
        mapped = result[column].map(_as_bool)
        if mapped.isna().any():
            raise ValueError(f"{column} must contain true/false or 1/0 values.")
        result[column] = mapped.astype(bool)
    if result[["vulnerability_id", "candidate_id"]].isna().any().any() or any(
        result[column].map(lambda value: isinstance(value, str) and not value.strip()).any()
        for column in ("vulnerability_id", "candidate_id")
    ):
        raise ValueError("Vulnerability and candidate identifiers cannot be blank.")
    if result.duplicated(["vulnerability_id", "candidate_id"]).any():
        raise ValueError("Candidate IDs must be unique within each vulnerability.")
    return result


def _as_bool(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value in (1, 1.0, "1"):
        return True
    if value in (0, 0.0, "0"):
        return False
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "y"}:
            return True
        if normalized in {"false", "no", "n"}:
            return False
    return None
