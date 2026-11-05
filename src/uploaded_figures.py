"""Generate exploratory figures and label-driven tuning from uploaded candidates."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.fuzzygen_bpd import BREAKPOINTS, FEATURES, SINGLETONS, trapezoid


LABEL_COLUMN = "patch_acceptable"
OPTIMIZER_BUDGET = 50
METHODS = ("Random Search", "Genetic Algorithm")
COLORS = {"Random Search": "#176b87", "Genetic Algorithm": "#c27b31"}


def validate_acceptability_labels(data: pd.DataFrame) -> pd.DataFrame:
    """Require a binary patch-acceptability label for uploaded-data optimization."""
    if LABEL_COLUMN not in data:
        raise ValueError(
            f"Uploaded CSV must include a `{LABEL_COLUMN}` column with true/false or 1/0 values."
        )
    result = data.copy()
    if result.empty:
        raise ValueError("Uploaded CSV must contain at least one candidate row.")
    labels = result[LABEL_COLUMN].map(_as_label)
    if labels.isna().any():
        raise ValueError(
            f"`{LABEL_COLUMN}` must contain only true/false or 1/0 values."
        )
    result[LABEL_COLUMN] = labels.astype(int)
    return result


def generate_uploaded_figures(
    results: pd.DataFrame,
    current_weights: tuple[float, float, float],
) -> dict[str, tuple[str, go.Figure, int]]:
    """Build the figure collection from uploaded candidate rows and their labels."""
    data = validate_acceptability_labels(results)
    eligible = data.loc[data["eligible"]].copy().reset_index(drop=True)
    if eligible.empty or eligible[LABEL_COLUMN].nunique() != 2:
        raise ValueError(
            "Uploaded data needs eligible candidates from both `patch_acceptable` classes "
            "to generate the complete labeled figure collection."
        )

    labels = eligible[LABEL_COLUMN].to_numpy(dtype=int)
    feature_scores = eligible[[f"F_{feature}" for feature in FEATURES]].to_numpy(dtype=float)
    folds = _cross_validation_splits(eligible)
    optimization = _optimize(feature_scores, labels, folds)

    figures: dict[str, tuple[str, go.Figure, int]] = {}
    _add_dataset_profile(figures, results)
    _add_membership_profiles(figures, results)
    _add_inference_figure(figures, eligible, feature_scores, optimization)
    _add_weight_ranking_figure(
        figures, eligible, feature_scores, optimization, current_weights
    )
    _add_convergence_figure(figures, optimization)
    _add_optimized_weight_figure(figures, optimization)
    _add_consistency_figure(figures, feature_scores, labels, folds, optimization)
    _add_ablation_figure(figures, feature_scores, labels, folds, optimization)
    _add_feature_distribution_figure(figures, results)
    _add_quality_figure(figures, eligible, feature_scores, optimization)
    _add_strategy_ranking_figure(figures, eligible, feature_scores, optimization)
    _add_fold_performance_figure(figures, labels, folds, optimization)
    _add_membership_sensitivity_figure(figures, eligible, feature_scores, optimization)
    _add_macro_f1_figure(figures, optimization)
    return figures


def _as_label(value):
    if pd.isna(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if value in (1, 1.0, "1"):
        return 1
    if value in (0, 0.0, "0"):
        return 0
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "y"}:
            return 1
        if normalized in {"false", "no", "n"}:
            return 0
    return None


def _cross_validation_splits(data: pd.DataFrame) -> list[np.ndarray]:
    groups = data["vulnerability_id"].astype(str).to_numpy()
    unique_groups = np.unique(groups)
    if len(unique_groups) > 1:
        rng = np.random.default_rng(42)
        rng.shuffle(unique_groups)
        buckets = [[] for _ in range(min(5, len(unique_groups)))]
        for index, group in enumerate(unique_groups):
            buckets[index % len(buckets)].append(group)
        return [np.flatnonzero(np.isin(groups, bucket)) for bucket in buckets]

    labels = data[LABEL_COLUMN].to_numpy(dtype=int)
    class_indices = [np.flatnonzero(labels == label) for label in (0, 1)]
    fold_count = min(5, *(len(indices) for indices in class_indices))
    if fold_count < 2:
        raise ValueError(
            "Need at least two eligible candidates from each `patch_acceptable` class "
            "to generate optimizer figures."
        )
    rng = np.random.default_rng(42)
    buckets = [[] for _ in range(fold_count)]
    for indices in class_indices:
        shuffled = rng.permutation(indices)
        for index, candidate in enumerate(shuffled):
            buckets[index % fold_count].append(int(candidate))
    return [np.asarray(bucket, dtype=int) for bucket in buckets]


def _macro_f1(labels: np.ndarray, predictions: np.ndarray) -> float:
    class_scores = []
    for label in (0, 1):
        true_positive = np.sum((labels == label) & (predictions == label))
        false_positive = np.sum((labels != label) & (predictions == label))
        false_negative = np.sum((labels == label) & (predictions != label))
        denominator = 2 * true_positive + false_positive + false_negative
        class_scores.append(
            0.0 if denominator == 0 else 2 * true_positive / denominator
        )
    return float(np.mean(class_scores))


def _normalize_weights(parameters: np.ndarray) -> np.ndarray:
    weights = np.maximum(parameters[:3], 1e-8)
    return weights / weights.sum()


def _fold_scores(
    feature_scores: np.ndarray,
    labels: np.ndarray,
    folds: list[np.ndarray],
    weights: np.ndarray,
    threshold: float,
) -> tuple[list[float], np.ndarray]:
    candidate_scores = feature_scores @ weights
    predictions = candidate_scores < threshold
    fold_f1 = [_macro_f1(labels[indices], predictions[indices]) for indices in folds]
    return fold_f1, predictions.astype(int)


def _optimize(
    feature_scores: np.ndarray,
    labels: np.ndarray,
    folds: list[np.ndarray],
) -> dict:
    def evaluate(parameters: np.ndarray) -> float:
        threshold = 0.1 + 0.8 * float(parameters[3])
        values, _ = _fold_scores(
            feature_scores, labels, folds, _normalize_weights(parameters), threshold
        )
        return float(np.mean(values))

    rng = np.random.default_rng(2718)
    random_trace = []
    random_parameters = []
    random_scores = []
    best_score = -1.0
    for parameters in rng.random((OPTIMIZER_BUDGET, 4)):
        score = evaluate(parameters)
        random_parameters.append(parameters)
        random_scores.append(score)
        best_score = max(best_score, score)
        random_trace.append(best_score)

    genetic_trace = []
    population = [rng.random(4) for _ in range(10)]
    population_scores = []
    for parameters in population:
        score = evaluate(parameters)
        population_scores.append(score)
        genetic_trace.append(max(population_scores))
    while len(genetic_trace) < OPTIMIZER_BUDGET:
        elite_indices = np.argsort(population_scores)[-2:]
        parents = [population[index] for index in elite_indices]
        alpha = rng.uniform(-0.2, 1.2, size=4)
        child = np.clip(alpha * parents[0] + (1 - alpha) * parents[1], 0, 1)
        mutation = rng.random(4) < 0.35
        child[mutation] = np.clip(
            child[mutation] + rng.normal(0, 0.15, mutation.sum()), 0, 1
        )
        score = evaluate(child)
        replace_index = int(np.argmin(population_scores))
        if score >= population_scores[replace_index]:
            population[replace_index] = child
            population_scores[replace_index] = score
        genetic_trace.append(max(population_scores))

    optimizers = {}
    for name, parameters, trace in (
        ("Random Search", random_parameters[int(np.argmax(random_scores))], random_trace),
        ("Genetic Algorithm", population[int(np.argmax(population_scores))], genetic_trace),
    ):
        threshold = 0.1 + 0.8 * float(parameters[3])
        fold_f1, predictions = _fold_scores(
            feature_scores, labels, folds, _normalize_weights(parameters), threshold
        )
        optimizers[name] = {
            "weights": _normalize_weights(parameters),
            "threshold": threshold,
            "fold_f1": fold_f1,
            "mean_f1": float(np.mean(fold_f1)),
            "std_f1": float(np.std(fold_f1)),
            "predictions": predictions,
            "trace": trace,
        }

    baseline_weights = np.full(3, 1 / 3)
    baseline_results = []
    for threshold in np.linspace(0.1, 0.9, 17):
        fold_f1, _ = _fold_scores(
            feature_scores, labels, folds, baseline_weights, threshold
        )
        baseline_results.append((float(np.mean(fold_f1)), float(threshold), fold_f1))
    baseline_mean, baseline_threshold, baseline_folds = max(baseline_results)
    return {
        "optimizers": optimizers,
        "baseline": {
            "weights": baseline_weights,
            "threshold": baseline_threshold,
            "fold_f1": baseline_folds,
            "mean_f1": baseline_mean,
            "std_f1": float(np.std(baseline_folds)),
        },
        "folds": folds,
        "labels": labels,
    }


def _register(figures, name, description, figure, height=500):
    figures[name] = (description, figure, height)


def _add_dataset_profile(figures, data):
    counts = data.groupby("vulnerability_id", as_index=False).agg(
        Candidates=("candidate_id", "size"),
        Eligible=("eligible", "sum"),
    )
    counts["Blocked by gates"] = counts["Candidates"] - counts["Eligible"]
    melted = counts.melt(
        id_vars="vulnerability_id",
        value_vars=["Eligible", "Blocked by gates"],
        var_name="Gate outcome",
        value_name="Candidate count",
    )
    figure = px.bar(
        melted,
        x="vulnerability_id",
        y="Candidate count",
        color="Gate outcome",
        barmode="stack",
        title="Uploaded CSV profile: candidates and gate outcomes",
        color_discrete_map={"Eligible": "#176b87", "Blocked by gates": "#c27b31"},
    )
    figure.update_layout(template="plotly_white", height=480, xaxis_title="Vulnerability")
    _register(
        figures,
        "Fig. 1 · Uploaded CSV Profile",
        "Candidate counts and gate outcomes from the uploaded file.",
        figure,
        480,
    )


def _add_membership_profiles(figures, data):
    figure = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=FEATURES,
        horizontal_spacing=0.08,
    )
    grid = np.linspace(0, 1, 201)
    terms = ("Low / Small", "Medium", "High / Large")
    colors = ("#176b87", "#c27b31", "#65499b")
    for column, feature in enumerate(FEATURES, 1):
        for term_index, points in enumerate(BREAKPOINTS[feature]):
            membership = [trapezoid(value, *points) for value in grid]
            figure.add_trace(
                go.Scatter(
                    x=grid,
                    y=membership,
                    mode="lines",
                    name=terms[term_index],
                    legendgroup=terms[term_index],
                    showlegend=column == 1,
                    line={"color": colors[term_index]},
                ),
                row=1,
                col=column,
            )
        for label, color in ((0, "#c27b31"), (1, "#176b87")):
            labeled = data[data[LABEL_COLUMN] == label]
            figure.add_trace(
                go.Scatter(
                    x=labeled[feature],
                    y=[0.02] * len(labeled),
                    mode="markers",
                    name="Unacceptable" if label == 0 else "Acceptable",
                    legendgroup=f"label-{label}",
                    showlegend=column == 1,
                    marker={"color": color, "symbol": "x", "size": 8},
                    customdata=labeled[["candidate_id", LABEL_COLUMN]].to_numpy(),
                    hovertemplate="Candidate: %{customdata[0]}<br>Feature: %{x:.3f}<extra></extra>",
                ),
                row=1,
                col=column,
            )
        figure.update_xaxes(title_text="Uploaded feature value", range=[0, 1], row=1, col=column)
        figure.update_yaxes(title_text="Membership degree", range=[0, 1], row=1, col=column)
    figure.update_layout(
        template="plotly_white",
        title="Fuzzy membership profiles with uploaded candidate values",
        height=500,
    )
    _register(
        figures,
        "Fig. 2 · Fuzzy Membership Profiles of Patch Disturbance Features",
        "Paper membership curves with the uploaded candidate feature values overlaid.",
        figure,
    )


def _add_inference_figure(figures, data, feature_scores, optimization):
    best = optimization["optimizers"]["Genetic Algorithm"]
    bpd = feature_scores @ best["weights"]
    plotted = data.copy()
    plotted["BPD"] = bpd
    plotted["Fuzzy label"] = plotted[LABEL_COLUMN].map(
        {0: "Unacceptable", 1: "Acceptable"}
    )
    figure = px.scatter_3d(
        plotted,
        x="F_TestDelta",
        y="F_CoverageDelta",
        z="F_DiffSize",
        color="BPD",
        symbol="Fuzzy label",
        hover_name="candidate_id",
        hover_data=["vulnerability_id", LABEL_COLUMN],
        color_continuous_scale="Teal",
        title="Uploaded candidates through fuzzification and tuned Sugeno scoring",
        labels={
            "F_TestDelta": "Fuzzy TestDelta",
            "F_CoverageDelta": "Fuzzy CoverageDelta",
            "F_DiffSize": "Fuzzy DiffSize",
        },
    )
    figure.update_layout(template="plotly_white", height=560)
    _register(
        figures,
        "Fig. 3 · Fuzzy Inference and Disturbance Score Transformation",
        "Fuzzified uploaded records and their genetic-algorithm-tuned BPD scores.",
        figure,
        560,
    )


def _add_weight_ranking_figure(
    figures, data, feature_scores, optimization, current_weights
):
    weights_by_method = {
        "Equal weights": np.full(3, 1 / 3),
        **{
            name: value["weights"]
            for name, value in optimization["optimizers"].items()
        },
        "Current UI weights": np.asarray(current_weights, dtype=float)
        / np.sum(current_weights),
    }
    chart_rows = []
    for method, weights in weights_by_method.items():
        scores = feature_scores @ weights
        method_rows = data[["vulnerability_id", "candidate_id"]].copy()
        method_rows["Method"] = method
        method_rows["Rank"] = (
            pd.Series(scores)
            .groupby(data["vulnerability_id"].reset_index(drop=True))
            .rank(method="min")
            .astype(int)
            .to_numpy()
        )
        chart_rows.append(method_rows)
    plotted = pd.concat(chart_rows, ignore_index=True)
    plotted["Candidate"] = (
        plotted["vulnerability_id"].astype(str)
        + " / "
        + plotted["candidate_id"].astype(str)
    )
    figure = px.line(
        plotted,
        x="Method",
        y="Rank",
        color="Candidate",
        line_group="Candidate",
        markers=True,
        hover_data=["vulnerability_id", "candidate_id"],
        title="Uploaded candidate ranking under alternative feature weights",
    )
    figure.update_yaxes(autorange="reversed", title="Rank within vulnerability")
    figure.update_layout(template="plotly_white", height=560, legend_title_text="Candidate")
    _register(
        figures,
        "Fig. 4 · Impact of Feature Weights on Candidate Patch Ranking",
        "Ranks change across equal, optimized, and current UI weights.",
        figure,
        560,
    )


def _add_convergence_figure(figures, optimization):
    rows = []
    for method, values in optimization["optimizers"].items():
        rows.extend(
            {"Evaluation": index + 1, "Best Macro-F1": score, "Algorithm": method}
            for index, score in enumerate(values["trace"])
        )
    figure = px.line(
        pd.DataFrame(rows),
        x="Evaluation",
        y="Best Macro-F1",
        color="Algorithm",
        color_discrete_map=COLORS,
        title="Label-driven optimizer convergence on uploaded candidates",
    )
    figure.update_layout(template="plotly_white", height=500, yaxis_range=[0, 1])
    _register(
        figures,
        "Fig. 5 · Convergence Behaviour of Hyperparameter Optimization Algorithms",
        "Best cross-validated Macro-F1 by evaluation; optimization uses uploaded labels.",
        figure,
    )


def _add_optimized_weight_figure(figures, optimization):
    rows = [
        {"Algorithm": "Equal weights", "Feature": feature, "Weight": 1 / len(FEATURES)}
        for feature in FEATURES
    ]
    for method, values in optimization["optimizers"].items():
        rows.extend(
            {"Algorithm": method, "Feature": feature, "Weight": weight}
            for feature, weight in zip(FEATURES, values["weights"])
        )
    figure = px.bar(
        pd.DataFrame(rows),
        x="Algorithm",
        y="Weight",
        color="Feature",
        barmode="group",
        text_auto=".2f",
        title="Feature weights optimized against uploaded labels",
        color_discrete_sequence=["#176b87", "#c27b31", "#65499b"],
    )
    figure.update_layout(template="plotly_white", height=500, yaxis_range=[0, 1])
    _register(
        figures,
        "Fig. 6 · Comparison of Optimized Feature Weights Across Algorithms",
        "Random-search and genetic-algorithm weights compared with an equal-weight baseline.",
        figure,
    )


def _add_consistency_figure(figures, feature_scores, labels, folds, optimization):
    rows = []
    thresholds = np.linspace(0.1, 0.9, 17)
    for method, values in optimization["optimizers"].items():
        for threshold in thresholds:
            scores = feature_scores @ values["weights"]
            predictions = scores < threshold
            fold_scores = [
                _macro_f1(labels[indices], predictions[indices]) for indices in folds
            ]
            rows.append(
                {
                    "Algorithm": method,
                    "Threshold": threshold,
                    "Mean fold Macro-F1": np.mean(fold_scores),
                    "Fold Macro-F1 SD": np.std(fold_scores),
                }
            )
    figure = px.scatter(
        pd.DataFrame(rows),
        x="Fold Macro-F1 SD",
        y="Mean fold Macro-F1",
        color="Algorithm",
        symbol="Algorithm",
        color_discrete_map=COLORS,
        hover_data=["Threshold"],
        title="Uploaded-label performance versus fold consistency",
    )
    figure.update_layout(template="plotly_white", height=520, yaxis_range=[0, 1])
    _register(
        figures,
        "Fig. 7 · Performance–Consistency Trade-off in Weight Optimization",
        "Each point is a threshold; lower fold Macro-F1 SD indicates more consistent folds.",
        figure,
        520,
    )


def _add_ablation_figure(figures, feature_scores, labels, folds, optimization):
    best = optimization["optimizers"]["Genetic Algorithm"]
    rows = []
    for omitted, feature in enumerate(FEATURES):
        weights = best["weights"].copy()
        weights[omitted] = 0
        weights = weights / weights.sum() if weights.sum() else np.full(3, 1 / 3)
        best_score = -1
        best_threshold = 0.5
        for threshold in np.linspace(0.1, 0.9, 17):
            fold_values, _ = _fold_scores(feature_scores, labels, folds, weights, threshold)
            mean_score = float(np.mean(fold_values))
            if mean_score > best_score:
                best_score, best_threshold = mean_score, threshold
        rows.append(
            {"Omitted feature": feature, "Mean fold Macro-F1": best_score, "Threshold": best_threshold}
        )
    figure = px.bar(
        pd.DataFrame(rows),
        x="Omitted feature",
        y="Mean fold Macro-F1",
        color="Omitted feature",
        text_auto=".3f",
        title="Uploaded-label feature ablation",
        hover_data=["Threshold"],
    )
    figure.update_layout(template="plotly_white", height=480, showlegend=False, yaxis_range=[0, 1])
    _register(
        figures,
        "Fig. 8 · Ablation Analysis of FuzzyGen-BPD Components",
        "Cross-validated Macro-F1 after removing each feature from tuned weights.",
        figure,
        480,
    )


def _add_feature_distribution_figure(figures, data):
    long = data.melt(
        id_vars=[LABEL_COLUMN],
        value_vars=list(FEATURES),
        var_name="Feature",
        value_name="Value",
    )
    long["Acceptability"] = long[LABEL_COLUMN].map(
        {0: "Unacceptable", 1: "Acceptable"}
    )
    figure = px.histogram(
        long,
        x="Value",
        color="Acceptability",
        facet_col="Feature",
        barmode="overlay",
        nbins=20,
        opacity=0.7,
        title="Uploaded feature distributions by patch acceptability",
        color_discrete_map={"Unacceptable": "#c27b31", "Acceptable": "#176b87"},
    )
    figure.update_layout(template="plotly_white", height=460)
    _register(
        figures,
        "Additional · Distribution of Patch Disturbance Features",
        "Uploaded TestDelta, CoverageDelta, and DiffSize values grouped by label.",
        figure,
        460,
    )


def _add_quality_figure(figures, data, feature_scores, optimization):
    best = optimization["optimizers"]["Genetic Algorithm"]
    plotted = data.copy()
    plotted["BPD"] = feature_scores @ best["weights"]
    plotted["Acceptability"] = plotted[LABEL_COLUMN].map(
        {0: "Unacceptable", 1: "Acceptable"}
    )
    figure = px.box(
        plotted,
        x="Acceptability",
        y="BPD",
        points="all",
        color="Acceptability",
        title="Tuned BPD scores for uploaded candidates by acceptability label",
        color_discrete_map={"Unacceptable": "#c27b31", "Acceptable": "#176b87"},
        hover_data=["vulnerability_id", "candidate_id"],
    )
    figure.update_layout(template="plotly_white", height=480, showlegend=False)
    _register(
        figures,
        "Additional · Relationship Between BPD Scores and Patch Quality",
        "Genetic-algorithm BPD scores computed from the uploaded feature values and labels.",
        figure,
        480,
    )


def _add_strategy_ranking_figure(figures, data, feature_scores, optimization):
    weights = {
        name: values["weights"]
        for name, values in optimization["optimizers"].items()
    }
    weights["Equal-weight baseline"] = np.full(3, 1 / 3)
    rows = []
    for method, method_weights in weights.items():
        ranks = (
            pd.Series(feature_scores @ method_weights)
            .groupby(data["vulnerability_id"].reset_index(drop=True))
            .rank(method="min")
            .astype(int)
        )
        for index, row in data.iterrows():
            rows.append(
                {
                    "Vulnerability": row["vulnerability_id"],
                    "Candidate": row["candidate_id"],
                    "Method": method,
                    "Rank": int(ranks.iloc[index]),
                }
            )
    figure = px.bar(
        pd.DataFrame(rows),
        x="Candidate",
        y="Rank",
        color="Method",
        barmode="group",
        facet_col="Vulnerability",
        title="Uploaded candidates ranked by each optimized strategy",
        color_discrete_map={**COLORS, "Equal-weight baseline": "#65499b"},
    )
    figure.update_yaxes(autorange="reversed")
    figure.update_layout(template="plotly_white", height=480)
    _register(
        figures,
        "Additional · Candidate Patch Ranking Across Optimization Strategies",
        "Candidate ranks from the uploaded records under each tuned weight set.",
        figure,
        480,
    )


def _add_fold_performance_figure(figures, labels, folds, optimization):
    rows = []
    for method, values in optimization["optimizers"].items():
        for index, (fold, score) in enumerate(zip(folds, values["fold_f1"]), 1):
            rows.append(
                {
                    "Method": method,
                    "Validation fold": index,
                    "Macro-F1": score,
                    "Candidates": len(fold),
                }
            )
    figure = px.box(
        pd.DataFrame(rows),
        x="Method",
        y="Macro-F1",
        color="Method",
        points="all",
        color_discrete_map=COLORS,
        title="Cross-validation performance on uploaded candidates",
    )
    figure.update_layout(template="plotly_white", height=470, yaxis_range=[0, 1])
    _register(
        figures,
        "Additional · Held-Out Performance Across Vulnerability Programs",
        "Fold-level Macro-F1; vulnerabilities are grouped when multiple groups are available.",
        figure,
        470,
    )


def _add_membership_sensitivity_figure(figures, data, feature_scores, optimization):
    base_weights = optimization["optimizers"]["Genetic Algorithm"]["weights"]
    original = feature_scores @ base_weights
    rows = []
    for scale in (0.9, 1.0, 1.1):
        changed = np.empty_like(feature_scores)
        for column, feature in enumerate(FEATURES):
            points_by_term = BREAKPOINTS[feature]
            values = data[feature].to_numpy(dtype=float)
            memberships = np.column_stack(
                [
                    [
                        trapezoid(
                            float(value),
                            *_scale_breakpoints(points, scale),
                        )
                        for value in values
                    ]
                    for points in points_by_term
                ]
            )
            changed[:, column] = (
                memberships @ np.asarray(SINGLETONS)
            ) / memberships.sum(axis=1)
        scores = changed @ base_weights
        for index, row in data.iterrows():
            rows.append(
                {
                    "Candidate": row["candidate_id"],
                    "Breakpoint scale": f"{scale:.1f}×",
                    "BPD change": float(scores[index] - original[index]),
                }
            )
    figure = px.box(
        pd.DataFrame(rows),
        x="Breakpoint scale",
        y="BPD change",
        points="all",
        color="Breakpoint scale",
        title="Uploaded candidate sensitivity to membership breakpoints",
    )
    figure.update_layout(template="plotly_white", height=480, showlegend=False)
    _register(
        figures,
        "Additional · Sensitivity of BPD Scores to Membership Function Parameters",
        "Score changes for uploaded rows when membership breakpoints are scaled by ±10%.",
        figure,
        480,
    )


def _scale_breakpoints(points, scale):
    center = (points[0] + points[-1]) / 2
    return tuple(float(np.clip(center + (point - center) * scale, 0, 1)) for point in points)


def _add_macro_f1_figure(figures, optimization):
    rows = [
        {"Optimizer": method, "Macro-F1": values["mean_f1"]}
        for method, values in optimization["optimizers"].items()
    ]
    rows.append(
        {
            "Optimizer": "Equal-weight baseline",
            "Macro-F1": optimization["baseline"]["mean_f1"],
        }
    )
    figure = px.bar(
        pd.DataFrame(rows),
        x="Optimizer",
        y="Macro-F1",
        color="Optimizer",
        text_auto=".3f",
        title="Cross-validated Macro-F1 across uploaded-data optimization strategies",
        color_discrete_map={**COLORS, "Equal-weight baseline": "#65499b"},
    )
    figure.update_layout(template="plotly_white", height=470, showlegend=False, yaxis_range=[0, 1])
    _register(
        figures,
        "Additional · Macro-F1 Performance Comparison Across Optimization Strategies",
        "Mean validation-fold Macro-F1 for uploaded-label optimizer searches.",
        figure,
        470,
    )
