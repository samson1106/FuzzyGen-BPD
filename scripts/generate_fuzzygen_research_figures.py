"""Create the research-figure set for the synthetic FuzzyGen-BPD demonstration.

All analytical figures use synthetic data defined in the conference-paper draft.
They validate the pipeline and visualise its behavior; they are not PatchEval or
Vul4J benchmark claims.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from run_fuzzygen_bpd_pipeline import (
    MFS, OUT, ROOT, SINGLETONS, bpd_matrix, fuzzy_feature, groups_to_folds,
    macro_f1, ranking_metrics, synthesize, trap,
)

PALETTE = {
    "TestDelta": "#176b87", "CoverageDelta": "#c27b31", "DiffSize": "#65499b",
    "Random Search": "#176b87", "GA": "#c27b31", "TPE": "#65499b",
    "GP-BO": "#2f855a", "SMAC": "#cf5a4a", "Equal-weight BPD": "#607d8b",
    "Jmean": "#176b87", "Jcons": "#c27b31",
}
METHOD_ORDER = ["Random Search", "GA", "TPE", "GP-BO", "SMAC", "Equal-weight BPD"]


def save(fig: go.Figure, filename: str) -> None:
    fig.update_layout(
        template="plotly_white", font=dict(family="Arial, sans-serif", size=14, color="#1f2937"),
        paper_bgcolor="white", plot_bgcolor="white", margin=dict(l=70, r=35, t=85, b=70),
    )
    fig.write_html(str(OUT / filename), include_plotlyjs="cdn")


def figure_2_memberships() -> None:
    x = np.linspace(0, 1, 500)
    fig = make_subplots(rows=1, cols=3, subplot_titles=["TestDelta", "CoverageDelta", "DiffSize"], shared_yaxes=True)
    terms = [("Low / Small", "#176b87"), ("Medium", "#c27b31"), ("High / Large", "#2f855a")]
    for col, (feature, profiles) in enumerate(MFS.items(), 1):
        for idx, ((term, color), points) in enumerate(zip(terms, profiles)):
            fig.add_trace(go.Scatter(x=x, y=trap(x, *points), mode="lines", name=term,
                                     line=dict(color=color, width=3), showlegend=col == 1), row=1, col=col)
        fig.update_xaxes(title_text="Feature value", range=[0, 1], row=1, col=col)
    fig.update_yaxes(title_text="Membership degree", range=[0, 1.05], row=1, col=1)
    fig.update_layout(title="Fig. 2 · Fuzzy Membership Profiles of Patch Disturbance Features", legend_title_text="Linguistic term", height=420)
    save(fig, "fig_02_membership_profiles.html")


def figure_3_transformation() -> None:
    x = np.linspace(0, 1, 500)
    fig = make_subplots(rows=1, cols=2, subplot_titles=["TestDelta fuzzification", "Sugeno disturbance transformation"])
    for (term, color), points in zip([("Low", "#176b87"), ("Medium", "#c27b31"), ("High", "#2f855a")], MFS["TestDelta"]):
        fig.add_trace(go.Scatter(x=x, y=trap(x, *points), mode="lines", name=term, line=dict(color=color, width=3)), row=1, col=1)
    transformed = fuzzy_feature(x, "TestDelta")
    fig.add_trace(go.Scatter(x=x, y=transformed, mode="lines", name="F(TestDelta)", line=dict(color="#65499b", width=4)), row=1, col=2)
    for y, label in zip(SINGLETONS, ["Low = 0.1", "Medium = 0.5", "High = 0.9"]):
        fig.add_hline(y=float(y), line_dash="dot", line_color="#94a3b8", annotation_text=label, annotation_position="bottom right", row=1, col=2)
    fig.update_xaxes(title_text="TestDelta", range=[0, 1])
    fig.update_yaxes(title_text="Membership degree", range=[0, 1.05], row=1, col=1)
    fig.update_yaxes(title_text="Fuzzy disturbance F(T)", range=[0, 1], row=1, col=2)
    fig.update_layout(title="Fig. 3 · Fuzzy Inference and Disturbance Score Transformation", height=430)
    save(fig, "fig_03_inference_transformation.html")


def figure_4_weight_impact() -> None:
    candidates = pd.DataFrame([
        ("A", .22, .035, .10), ("B", .08, .060, .38), ("C", .40, .012, .04), ("D", .55, .120, .20),
    ], columns=["Candidate", "TestDelta", "CoverageDelta", "DiffSize"])
    for feature in MFS:
        candidates[f"F_{feature}"] = fuzzy_feature(candidates[feature].to_numpy(), feature)
    profiles = {
        "Equal weights": np.array([1/3, 1/3, 1/3]),
        "Test-focused": np.array([.6, .2, .2]),
        "Coverage-focused": np.array([.2, .6, .2]),
        "Diff-focused": np.array([.2, .2, .6]),
    }
    rows = []
    value_matrix = candidates[["F_TestDelta", "F_CoverageDelta", "F_DiffSize"]].to_numpy()
    for profile, weights in profiles.items():
        for candidate, score in zip(candidates.Candidate, value_matrix @ weights):
            rows.append({"Weighting": profile, "Candidate": candidate, "BPD": score})
    frame = pd.DataFrame(rows)
    fig = px.bar(frame, x="Weighting", y="BPD", color="Candidate", barmode="group", text_auto=".3f",
                 category_orders={"Weighting": list(profiles)}, color_discrete_map={"A":"#176b87","B":"#c27b31","C":"#2f855a","D":"#65499b"})
    fig.update_yaxes(title="Behavioral Patch Distance", range=[0, .75])
    fig.update_layout(title="Fig. 4 · Impact of Feature Weights on Candidate Patch Ranking", height=470, legend_title_text="Candidate")
    save(fig, "fig_04_weight_impact.html")


def figure_5_convergence(traces: pd.DataFrame) -> None:
    summary = traces.groupby(["method", "evaluation"], as_index=False).agg(
        score=("best_inner_macro_f1", "mean"), std=("best_inner_macro_f1", "std")
    )
    fig = go.Figure()
    for method in [m for m in METHOD_ORDER if m != "Equal-weight BPD"]:
        part = summary[summary.method == method]
        fig.add_trace(go.Scatter(x=part.evaluation, y=part.score, mode="lines", name=method, line=dict(color=PALETTE[method], width=3)))
    fig.update_xaxes(title="Objective evaluations", range=[1, 60])
    fig.update_yaxes(title="Mean best inner-fold Macro-F1", range=[.5, 1])
    fig.update_layout(title="Fig. 5 · Convergence Behaviour of Hyperparameter Optimization Algorithms", height=480, legend_title_text="Optimizer")
    save(fig, "fig_05_optimizer_convergence.html")


def figure_6_weights(weights: pd.DataFrame) -> None:
    data = weights[weights.method != "Equal-weight BPD"].melt(id_vars="method", value_vars=["w_testdelta", "w_coveragedelta", "w_diffsize"], var_name="Feature", value_name="Weight")
    data["Feature"] = data.Feature.map({"w_testdelta":"TestDelta", "w_coveragedelta":"CoverageDelta", "w_diffsize":"DiffSize"})
    summary = data.groupby(["method", "Feature"], as_index=False).Weight.mean()
    fig = px.bar(summary, x="method", y="Weight", color="Feature", barmode="stack", text_auto=".2f",
                 category_orders={"method": [m for m in METHOD_ORDER if m != "Equal-weight BPD"]}, color_discrete_map={k: PALETTE[k] for k in ["TestDelta", "CoverageDelta", "DiffSize"]})
    fig.update_yaxes(title="Mean learned weight", range=[0, 1])
    fig.update_layout(title="Fig. 6 · Comparison of Optimized Feature Weights Across Algorithms", height=470, legend_title_text="Feature")
    save(fig, "fig_06_learned_weights.html")


def tune_threshold(scores: np.ndarray, labels: np.ndarray, inner: list[np.ndarray], consistency: bool) -> tuple[float, float, float]:
    best = (-np.inf, .5, np.nan)
    for threshold in np.linspace(.1, .9, 33):
        f1 = np.array([macro_f1(labels[idx], (scores[idx] < threshold).astype(int)) for idx in inner])
        score = f1.mean() - (.5 * f1.std(ddof=0) if consistency else 0)
        if score > best[0]:
            best = (float(score), float(threshold), float(f1.std(ddof=0)))
    return best


def consistency_analysis(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for seed in range(3):
        outer = groups_to_folds(data.program_id.to_numpy(), 5, np.random.default_rng(100 + seed))
        for fold, test_idx in enumerate(outer, 1):
            train_idx = np.setdiff1d(np.arange(len(data)), test_idx)
            train, test = data.iloc[train_idx].reset_index(drop=True), data.iloc[test_idx]
            inner = groups_to_folds(train.program_id.to_numpy(), 4, np.random.default_rng(200 + seed*10 + fold))
            scores_train = bpd_matrix(train) @ np.array([1/3, 1/3, 1/3])
            scores_test = bpd_matrix(test) @ np.array([1/3, 1/3, 1/3])
            labels = train.acceptable.to_numpy(dtype=int)
            for objective in ("Jmean", "Jcons"):
                _, threshold, inner_std = tune_threshold(scores_train, labels, inner, objective == "Jcons")
                result = macro_f1(test.acceptable.to_numpy(dtype=int), (scores_test < threshold).astype(int))
                rows.append({"seed": seed, "fold": fold, "objective": objective, "heldout_macro_f1": result, "inner_fold_sd": inner_std, "threshold": threshold})
    return pd.DataFrame(rows)


def figure_7_consistency(data: pd.DataFrame) -> pd.DataFrame:
    result = consistency_analysis(data)
    left = px.box(result, x="objective", y="inner_fold_sd", color="objective", points="all", color_discrete_map={k: PALETTE[k] for k in ["Jmean", "Jcons"]}, title="Inner-fold Macro-F1 dispersion")
    right = px.box(result, x="objective", y="heldout_macro_f1", color="objective", points="all", color_discrete_map={k: PALETTE[k] for k in ["Jmean", "Jcons"]}, title="Held-out Macro-F1")
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Inner-fold dispersion", "Held-out classification quality"])
    for source, col in ((left, 1), (right, 2)):
        for trace in source.data:
            trace.showlegend = col == 1
            fig.add_trace(trace, row=1, col=col)
    fig.update_yaxes(title_text="Standard deviation", row=1, col=1)
    fig.update_yaxes(title_text="Macro-F1", row=1, col=2, range=[.4, 1])
    fig.update_layout(title="Fig. 7 · Performance–Consistency Trade-off in Weight Optimization", height=470, legend_title_text="Objective")
    save(fig, "fig_07_performance_consistency.html")
    result.to_csv(OUT / "consistency_objective_metrics.csv", index=False)
    return result


def ablation_analysis(data: pd.DataFrame) -> pd.DataFrame:
    variants = {
        "All features": ("TestDelta", "CoverageDelta", "DiffSize"), "TestDelta only": ("TestDelta",),
        "CoverageDelta only": ("CoverageDelta",), "DiffSize only": ("DiffSize",),
        "Without TestDelta": ("CoverageDelta", "DiffSize"), "Without CoverageDelta": ("TestDelta", "DiffSize"),
        "Without DiffSize": ("TestDelta", "CoverageDelta"),
    }
    rows = []
    for seed in range(3):
        outer = groups_to_folds(data.program_id.to_numpy(), 5, np.random.default_rng(300 + seed))
        for fold, test_idx in enumerate(outer, 1):
            train_idx = np.setdiff1d(np.arange(len(data)), test_idx)
            train, test = data.iloc[train_idx].reset_index(drop=True), data.iloc[test_idx]
            inner = groups_to_folds(train.program_id.to_numpy(), 4, np.random.default_rng(400 + seed*10 + fold))
            for variant, selected in variants.items():
                cols = [f"F_{feature}" for feature in selected]
                tr_scores, te_scores = train[cols].mean(axis=1).to_numpy(), test[cols].mean(axis=1).to_numpy()
                _, threshold, _ = tune_threshold(tr_scores, train.acceptable.to_numpy(dtype=int), inner, False)
                rows.append({"seed":seed, "fold":fold, "variant":variant, "macro_f1":macro_f1(test.acceptable.to_numpy(dtype=int), (te_scores<threshold).astype(int))})
    return pd.DataFrame(rows)


def figure_8_ablation(data: pd.DataFrame) -> pd.DataFrame:
    result = ablation_analysis(data)
    order = ["All features", "Without TestDelta", "Without CoverageDelta", "Without DiffSize", "TestDelta only", "CoverageDelta only", "DiffSize only"]
    fig = px.box(result, x="variant", y="macro_f1", color="variant", points="all", category_orders={"variant":order},
                 color_discrete_sequence=["#176b87", "#c27b31", "#65499b", "#2f855a", "#cf5a4a", "#607d8b", "#9b59b6"])
    fig.update_layout(title="Fig. 8 · Ablation Analysis of FuzzyGen-BPD Components", height=490, showlegend=False)
    fig.update_xaxes(title="Feature configuration", tickangle=-25)
    fig.update_yaxes(title="Held-out Macro-F1", range=[.4, 1])
    save(fig, "fig_08_ablation.html")
    result.to_csv(OUT / "ablation_metrics.csv", index=False)
    return result


def additional_figures(data: pd.DataFrame, pred: pd.DataFrame, folds: pd.DataFrame) -> None:
    # Feature distribution small multiples by true synthetic label.
    long = data.melt(id_vars="acceptable", value_vars=list(MFS), var_name="Feature", value_name="Value")
    fig = px.histogram(long, x="Value", color="acceptable", facet_col="Feature", barmode="overlay", nbins=24,
                       color_discrete_map={0:"#c27b31",1:"#176b87"}, labels={"acceptable":"Synthetic label (1 = acceptable)"})
    fig.update_layout(title="Distribution of Patch Disturbance Features · synthetic candidates", height=420, legend_title_text="Synthetic label")
    fig.update_xaxes(range=[0,1])
    save(fig, "additional_01_feature_distributions.html")

    rs = pred[pred.method == "Random Search"].copy()
    fig = px.box(rs, x="acceptable", y="bpd", points="all", color="acceptable", color_discrete_map={0:"#c27b31",1:"#176b87"},
                 labels={"acceptable":"Synthetic patch quality (1 = acceptable)", "bpd":"Held-out BPD"})
    fig.update_layout(title="Relationship Between BPD Scores and Synthetic Patch Quality", height=440, showlegend=False)
    save(fig, "additional_02_bpd_quality.html")

    first_group = sorted(pred.rank_group.unique())[0]
    rank = pred[pred.rank_group == first_group].copy()
    fig = px.bar(rank, x="candidate_id", y="rank", color="method", barmode="group", text_auto=".0f", category_orders={"method":METHOD_ORDER}, color_discrete_map=PALETTE)
    fig.update_yaxes(title="Rank (1 = best)", autorange="reversed", dtick=1)
    fig.update_layout(title=f"Candidate Patch Ranking Across Optimization Strategies · {first_group}", height=460, legend_title_text="Method")
    save(fig, "additional_03_ranking_strategies.html")

    fig = px.strip(folds, x="method", y="macro_f1", color="seed", stripmode="overlay", category_orders={"method":METHOD_ORDER},
                   color_discrete_sequence=["#176b87", "#c27b31", "#65499b"], labels={"method":"Optimizer", "macro_f1":"Held-out Macro-F1"})
    fig.update_layout(title="Held-Out Performance Across Vulnerability Programs · grouped folds", height=460, legend_title_text="Split seed")
    fig.update_xaxes(tickangle=-20)
    save(fig, "additional_04_heldout_generalization.html")

    # Local sensitivity: vary all membership breakpoints around the original definitions while preserving shoulders.
    rows = []
    for scale in (.8, .9, 1., 1.1, 1.2):
        modified = {}
        for feature, profiles in MFS.items():
            shifted = []
            for a,b,c,d in profiles:
                # Move every non-shoulder interior breakpoint relative to zero, cap to the domain.
                shifted.append((a if a==0 else min(1,a*scale), b if b==0 else min(1,b*scale), c if c==1 else min(1,c*scale), d if d==1 else min(1,d*scale)))
            modified[feature] = tuple(shifted)
        fuzzy = np.column_stack([
            np.column_stack([trap(data[feature].to_numpy(), *points) for points in modified[feature]]) @ SINGLETONS /
            np.column_stack([trap(data[feature].to_numpy(), *points) for points in modified[feature]]).sum(axis=1)
            for feature in MFS
        ])
        scores = fuzzy.mean(axis=1)
        # A transparent global threshold sweep is used for this local robustness diagnostic only.
        f1 = max(macro_f1(data.acceptable.to_numpy(dtype=int), (scores<t).astype(int)) for t in np.linspace(.1,.9,81))
        rows.append({"Breakpoint scale":scale, "Best Macro-F1":f1})
    sensitivity = pd.DataFrame(rows)
    fig = px.line(sensitivity, x="Breakpoint scale", y="Best Macro-F1", markers=True, text="Best Macro-F1")
    fig.update_traces(line=dict(color="#176b87", width=3), marker=dict(size=9), texttemplate="%{text:.3f}", textposition="top center")
    fig.update_layout(title="Sensitivity of BPD Scores to Membership Function Parameters · synthetic diagnostic", height=440, showlegend=False)
    fig.update_yaxes(range=[.5, 1])
    save(fig, "additional_05_membership_sensitivity.html")
    sensitivity.to_csv(OUT / "membership_sensitivity.csv", index=False)

    summary = folds.groupby("method", as_index=False).macro_f1.mean().sort_values("macro_f1", ascending=False)
    fig = px.bar(summary, x="macro_f1", y="method", orientation="h", text_auto=".3f", color="method", category_orders={"method":list(summary.method[::-1])}, color_discrete_map=PALETTE)
    fig.update_layout(title="Macro-F1 Performance Comparison Across Optimization Strategies", height=450, showlegend=False)
    fig.update_xaxes(title="Mean held-out Macro-F1", range=[.5, .8])
    fig.update_yaxes(title="Optimizer")
    save(fig, "additional_06_macro_f1_comparison.html")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = synthesize()
    pred_path, folds_path, weights_path = OUT / "optimizer_heldout_predictions.csv", OUT / "optimizer_fold_metrics.csv", OUT / "optimizer_weights.csv"
    if not all(path.exists() for path in (pred_path, folds_path, weights_path)):
        raise FileNotFoundError("Run scripts/compare_fuzzygen_optimizers.py before generating the research figures.")
    pred, folds, weights = pd.read_csv(pred_path), pd.read_csv(folds_path), pd.read_csv(weights_path)
    figure_2_memberships()
    figure_3_transformation()
    figure_4_weight_impact()
    figure_5_convergence(pd.read_csv(OUT / "optimizer_search_traces.csv"))
    figure_6_weights(weights)
    consistency = figure_7_consistency(data)
    ablation = figure_8_ablation(data)
    additional_figures(data, pred, folds)
    manifest = {
        "status": "SYNTHETIC DEMONSTRATION ONLY; NOT PATCH EVAL OR VUL4J RESULTS",
        "figure_1": "User-supplied FuzzyGen-BPD Research Framework Infographic.png",
        "figures_generated": [f"Fig. {i}" for i in range(2, 9)],
        "additional_figures_generated": 6,
        "consistency_summary": consistency.groupby("objective")[["inner_fold_sd", "heldout_macro_f1"]].mean().to_dict(),
        "ablation_summary": ablation.groupby("variant").macro_f1.mean().sort_values(ascending=False).to_dict(),
    }
    (OUT / "research_figure_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
