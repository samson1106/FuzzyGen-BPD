"""Run the paper's synthetic, program-grouped FuzzyGen-BPD demonstration.

This is a reproducible pipeline demonstration, not PatchEval/Vul4J evidence.
It creates synthetic candidate features and labels from the distributions in
the paper, then evaluates fuzzy BPD on five held-out program folds.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/fuzzygen_bpd"
RNG_SEED = 2026
FOLDS = 5
WEIGHT_GRID = np.asarray([(i, j, 20 - i - j) for i in range(21) for j in range(21 - i)], dtype=float) / 20

MFS = {
    "TestDelta": ((0, 0, .15, .30), (.15, .30, .50, .65), (.50, .65, 1, 1)),
    "CoverageDelta": ((0, 0, .02, .05), (.02, .05, .10, .15), (.10, .15, 1, 1)),
    "DiffSize": ((0, 0, .05, .15), (.05, .15, .30, .45), (.30, .45, 1, 1)),
}
SINGLETONS = np.array([.1, .5, .9])


def trap(x, a, b, c, d):
    x = np.asarray(x, dtype=float)
    rise = np.where(x >= b, 1., 0.) if a == b else np.clip((x-a)/(b-a), 0, 1)
    fall = np.where(x <= c, 1., 0.) if c == d else np.clip((d-x)/(d-c), 0, 1)
    return np.maximum(0, np.minimum(np.minimum(rise, 1), fall))


def fuzzy_feature(x, feature):
    memberships = np.column_stack([trap(x, *points) for points in MFS[feature]])
    return memberships @ SINGLETONS / memberships.sum(axis=1)


def synthesize(seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    sizes = [5] * 2 + [4] * 58  # 242 candidates across 60 programs
    rng.shuffle(sizes)
    records = []
    label_id = np.repeat([0, 1], 121)
    rng.shuffle(label_id)
    cursor = 0
    for program_idx, n in enumerate(sizes):
        effect = rng.lognormal(mean=0, sigma=.22)
        for _ in range(n):
            label = int(label_id[cursor]); cursor += 1
            if label:
                t = rng.beta(1.3, 11)
                c = abs(rng.normal(.025, .02))
                d = rng.beta(1.6, 14)
            else:
                t = rng.beta(2.2, 6.5)
                c = abs(rng.normal(.05, .04))
                d = rng.beta(2, 9)
            t = min(1., t * effect)
            d = min(1., d * effect)
            records.append((f"SYN-P{program_idx+1:02}", f"C{sum(1 for row in records if row[0] == f'SYN-P{program_idx+1:02}')+1:02}", t, min(c, 1.), d, label))
    frame = pd.DataFrame(records, columns=["program_id", "candidate_id", "TestDelta", "CoverageDelta", "DiffSize", "acceptable"])
    # Model reviewer disagreement with nine balanced label swaps (18/242 = 7.4%).
    pos = rng.choice(frame.index[frame.acceptable == 1], 9, replace=False)
    neg = rng.choice(frame.index[frame.acceptable == 0], 9, replace=False)
    frame.loc[pos, "acceptable"] = 0
    frame.loc[neg, "acceptable"] = 1
    for name in MFS:
        frame[f"F_{name}"] = fuzzy_feature(frame[name].to_numpy(), name)
    return frame


def macro_f1(y, pred):
    values = []
    for cls in (0, 1):
        tp = np.sum((y == cls) & (pred == cls))
        fp = np.sum((y != cls) & (pred == cls))
        fn = np.sum((y == cls) & (pred != cls))
        denom = 2 * tp + fp + fn
        values.append(0. if denom == 0 else 2 * tp / denom)
    return float(np.mean(values))


def groups_to_folds(groups, k, rng):
    unique = np.array(sorted(set(groups)))
    rng.shuffle(unique)
    buckets = [[] for _ in range(k)]
    for i, group in enumerate(unique):
        buckets[i % k].append(group)
    return [np.flatnonzero(np.isin(groups, bucket)) for bucket in buckets]


def bpd_matrix(df):
    return df[["F_TestDelta", "F_CoverageDelta", "F_DiffSize"]].to_numpy()


def tune(train, rng, budget=60):
    groups = train.program_id.to_numpy()
    inner = groups_to_folds(groups, 4, rng)
    # Same four-dimensional search intent as the paper: simplex weights + threshold.
    idx = rng.choice(len(WEIGHT_GRID), size=min(budget, len(WEIGHT_GRID)), replace=False)
    thresholds = np.linspace(.10, .90, 17)
    x = bpd_matrix(train)
    y = train.acceptable.to_numpy(dtype=int)
    best = (-1., None, None)
    for wi in idx:
        w = WEIGHT_GRID[wi]
        score = x @ w
        for threshold in thresholds:
            fold_scores = []
            for val_idx in inner:
                pred = (score[val_idx] < threshold).astype(int)
                fold_scores.append(macro_f1(y[val_idx], pred))
            objective = float(np.mean(fold_scores))
            if objective > best[0]:
                best = (objective, w.copy(), float(threshold))
    return best[1], best[2], best[0]


def classification_metrics(y, pred):
    cm = np.zeros((2, 2), dtype=int)
    for truth, guess in zip(y, pred):
        cm[int(truth), int(guess)] += 1
    tn, fp, fn, tp = cm.ravel()
    return {"macro_f1": macro_f1(y, pred), "precision": float(tp/(tp+fp)) if tp+fp else 0.,
            "recall": float(tp/(tp+fn)) if tp+fn else 0., "specificity": float(tn/(tn+fp)) if tn+fp else 0.,
            "mcc": float(((tp*tn)-(fp*fn))/np.sqrt(max(1,(tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))),
            "confusion_matrix": cm.tolist()}


def ranking_metrics(frame, rank_col):
    top1, mrr, ndcg = [], [], []
    group_col = "rank_group" if "rank_group" in frame.columns else "program_id"
    for _, part in frame.sort_values(rank_col).groupby(group_col):
        labels = part.sort_values(rank_col).acceptable.to_numpy(dtype=int)
        relevant = np.flatnonzero(labels == 1)
        top1.append(float(labels[0] == 1))
        mrr.append(float(1 / (relevant[0] + 1)) if len(relevant) else 0.)
        discounts = 1 / np.log2(np.arange(2, len(labels)+2))
        ideal_labels = np.sort(labels)[::-1]
        ndcg.append(float(np.dot(labels, discounts) / np.dot(ideal_labels, discounts)) if ideal_labels.sum() else 0.)
    return {"top1": float(np.mean(top1)), "mrr": float(np.mean(mrr)), "ndcg": float(np.mean(ndcg))}


def cluster_bootstrap(frame, pred_col, rank_col, seed=77, reps=2000):
    rng = np.random.default_rng(seed)
    programs = frame.program_id.unique()
    metrics = {"macro_f1": [], "top1": [], "mrr": [], "ndcg": []}
    indexed = {p: frame[frame.program_id == p] for p in programs}
    confusion = []
    ranks = []
    for p in programs:
        part = indexed[p]
        cm = np.zeros((2,2), dtype=int)
        np.add.at(cm, (part.acceptable.to_numpy(dtype=int), part[pred_col].to_numpy(dtype=int)), 1)
        confusion.append(cm)
        ranks.append(ranking_metrics(part, rank_col))
    confusion = np.asarray(confusion)
    for _ in range(reps):
        sampled = rng.integers(0, len(programs), len(programs))
        cm = confusion[sampled].sum(axis=0)
        tn, fp, fn, tp = cm.ravel()
        f1_neg = 2*tn/(2*tn+fp+fn) if 2*tn+fp+fn else 0.
        f1_pos = 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.
        metrics["macro_f1"].append(float((f1_neg+f1_pos)/2))
        for name in ("top1", "mrr", "ndcg"):
            metrics[name].append(float(np.mean([ranks[i][name] for i in sampled])))
    return {name: [float(np.quantile(vals, .025)), float(np.quantile(vals, .975))] for name, vals in metrics.items()}


def paired_program_test(frame, reps=10000, seed=909):
    """Program-cluster bootstrap CI and paired sign-flip test vs equal weights."""
    per_program = []
    for program, part in frame.groupby("program_id"):
        y = part.acceptable.to_numpy(dtype=int)
        f_opt = macro_f1(y, part.pred_optimized.to_numpy(dtype=int))
        f_eq = macro_f1(y, part.pred_equal.to_numpy(dtype=int))
        top_opt = float(part.loc[part.rank_optimized == 1, "acceptable"].max())
        top_eq = float(part.loc[part.rank_equal == 1, "acceptable"].max())
        per_program.append((program, f_opt-f_eq, top_opt-top_eq))
    values = np.asarray([r[1:] for r in per_program], dtype=float)
    rng = np.random.default_rng(seed)
    observed = values.mean(axis=0)
    bootstrap = np.empty((reps, 2))
    for i in range(reps):
        bootstrap[i] = values[rng.integers(0, len(values), len(values))].mean(axis=0)
    signs = rng.choice([-1., 1.], size=(reps, len(values), 1))
    permuted = (values[None, :, :] * signs).mean(axis=1)
    p_values = ((np.abs(permuted) >= np.abs(observed)).sum(axis=0) + 1) / (reps + 1)
    return {name: {"mean_delta": float(observed[i]), "cluster_bootstrap_95_ci": [float(x) for x in np.quantile(bootstrap[:, i], [.025,.975])], "paired_sign_flip_p": float(p_values[i])}
            for i, name in enumerate(("macro_f1", "top1"))}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = synthesize()
    rng = np.random.default_rng(121)
    outer_folds = groups_to_folds(data.program_id.to_numpy(), FOLDS, rng)
    data["pred_optimized"] = -1
    data["pred_equal"] = -1
    fold_rows = []
    weight_rows = []
    for fold_id, test_idx in enumerate(outer_folds, start=1):
        train_idx = np.setdiff1d(np.arange(len(data)), test_idx)
        train = data.iloc[train_idx].reset_index(drop=True)
        test = data.iloc[test_idx].copy()
        w, threshold, inner_f1 = tune(train, np.random.default_rng(9000+fold_id))
        data.loc[test_idx, "bpd_optimized"] = bpd_matrix(test) @ w
        data.loc[test_idx, "pred_optimized"] = (data.loc[test_idx, "bpd_optimized"] < threshold).astype(int)
        eq_w = np.array([1/3]*3)
        # Fair equal-weight baseline: threshold tuned on training only using grouped inner folds.
        _, eq_threshold, _ = tune_equal(train, np.random.default_rng(10000+fold_id))
        data.loc[test_idx, "bpd_equal"] = bpd_matrix(test) @ eq_w
        data.loc[test_idx, "pred_equal"] = (data.loc[test_idx, "bpd_equal"] < eq_threshold).astype(int)
        test_y = test.acceptable.to_numpy(dtype=int)
        metrics = classification_metrics(test_y, data.loc[test_idx, "pred_optimized"].to_numpy(dtype=int))
        fold_rows.append({"fold": fold_id, "n_candidates": len(test), "n_programs": test.program_id.nunique(), "threshold": threshold, "inner_macro_f1": inner_f1, **{k:v for k,v in metrics.items() if k != "confusion_matrix"}})
        weight_rows.append({"fold": fold_id, "w_testdelta":w[0], "w_coveragedelta":w[1], "w_diffsize":w[2], "threshold":threshold})

    results = data.sort_values(["program_id", "candidate_id"]).reset_index(drop=True)
    results["pred_optimized"] = results.pred_optimized.astype(int)
    results["pred_equal"] = results.pred_equal.astype(int)
    y = results.acceptable.to_numpy(dtype=int)
    overall = classification_metrics(y, results.pred_optimized.to_numpy(dtype=int))
    equal = classification_metrics(y, results.pred_equal.to_numpy(dtype=int))
    results["rank_optimized"] = results.groupby("program_id").bpd_optimized.rank(method="min")
    results["rank_equal"] = results.groupby("program_id").bpd_equal.rank(method="min")
    rank_opt = ranking_metrics(results, "rank_optimized")
    rank_eq = ranking_metrics(results, "rank_equal")
    top1_opt, top1_eq = rank_opt["top1"], rank_eq["top1"]
    ci = cluster_bootstrap(results, "pred_optimized", "rank_optimized")
    equal_ci = cluster_bootstrap(results, "pred_equal", "rank_equal", seed=78)
    paired_test = paired_program_test(results)
    fold_df = pd.DataFrame(fold_rows)
    weights_df = pd.DataFrame(weight_rows)
    metrics = {
        "status": "SYNTHETIC DEMONSTRATION ONLY; NOT PATCH EVAL OR VUL4J RESULTS",
        "data_seed": RNG_SEED,
        "candidate_count": int(len(results)), "program_count": int(results.program_id.nunique()),
        "label_counts": {"unacceptable_0": int((y==0).sum()), "acceptable_1": int((y==1).sum())},
        "optimized": overall, "equal_weight_baseline": equal,
        "top1_accuracy_optimized": top1_opt, "top1_accuracy_equal_weight": top1_eq,
        "ranking_optimized": rank_opt, "ranking_equal_weight": rank_eq,
        "cluster_bootstrap_95_ci": ci,
        "equal_weight_cluster_bootstrap_95_ci": equal_ci,
        "paired_program_comparison_optimized_minus_equal": paired_test,
        "fold_macro_f1_mean": float(fold_df.macro_f1.mean()), "fold_macro_f1_sd": float(fold_df.macro_f1.std(ddof=1)),
        "folds": fold_rows, "weights_by_fold": weight_rows,
        "interpretation": "Labels and features are synthetically generated from the distributions documented in the paper. This run validates the pipeline mechanics only; it does not estimate real patch performance.",
    }
    results.to_csv(OUT/"synthetic_heldout_predictions.csv", index=False)
    fold_df.to_csv(OUT/"fold_metrics.csv", index=False)
    weights_df.to_csv(OUT/"learned_weights.csv", index=False)
    (OUT/"metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    make_charts(results, metrics)
    print(json.dumps(metrics, indent=2))


def tune_equal(train, rng):
    inner = groups_to_folds(train.program_id.to_numpy(), 4, rng)
    score = bpd_matrix(train) @ np.array([1/3]*3)
    y = train.acceptable.to_numpy(dtype=int)
    best = (-1., .5)
    for threshold in np.linspace(.1, .9, 17):
        values = [macro_f1(y[ix], (score[ix] < threshold).astype(int)) for ix in inner]
        if np.mean(values) > best[0]:
            best = (float(np.mean(values)), float(threshold))
    return np.array([1/3]*3), best[1], best[0]


def make_charts(df, metrics):
    cm = np.array(metrics["optimized"]["confusion_matrix"])
    fig = go.Figure(go.Heatmap(z=cm, x=["Predicted unacceptable", "Predicted acceptable"], y=["Actual unacceptable", "Actual acceptable"],
                               colorscale=[[0,"#eff6ff"],[1,"#145b8c"]], text=cm, texttemplate="%{text}", showscale=False,
                               hovertemplate="%{y}<br>%{x}<br>Count: %{z}<extra></extra>"))
    fig.update_layout(title="Held-out confusion matrix · synthetic demonstration", height=460, template="plotly_white",
                      xaxis_title="Prediction", yaxis_title="Ground truth", margin=dict(l=100,r=30,t=80,b=80),
                      annotations=[dict(text="Rows = actual class · columns = predicted class", xref="paper", yref="paper", x=.5, y=1.10, showarrow=False, font=dict(size=13,color="#516274"))])
    fig.write_html(str(OUT/"confusion_matrix.html"), include_plotlyjs="cdn")
    fig.write_image(str(OUT/"confusion_matrix.png")) if False else None
    fig2 = px.box(df, x="acceptable", y="bpd_optimized", points="all", color="acceptable",
                  color_discrete_map={0:"#c27b31",1:"#176b87"}, labels={"acceptable":"Actual label (0 = no, 1 = yes)","bpd_optimized":"Held-out BPD score"},
                  title="Held-out BPD score by synthetic acceptability label")
    fig2.update_layout(template="plotly_white", height=470, showlegend=False)
    fig2.write_html(str(OUT/"score_distributions.html"), include_plotlyjs="cdn")
    comparison = pd.DataFrame([
        {"model":"Optimized fuzzy BPD", "Macro-F1":metrics["optimized"]["macro_f1"], "MCC":metrics["optimized"]["mcc"], "Top-1":metrics["top1_accuracy_optimized"]},
        {"model":"Equal-weight fuzzy BPD", "Macro-F1":metrics["equal_weight_baseline"]["macro_f1"], "MCC":metrics["equal_weight_baseline"]["mcc"], "Top-1":metrics["top1_accuracy_equal_weight"]},
    ]).melt("model", var_name="Metric", value_name="Score")
    fig3 = px.bar(comparison, x="Metric", y="Score", color="model", barmode="group", text_auto=".3f",
                  color_discrete_map={"Optimized fuzzy BPD":"#176b87","Equal-weight fuzzy BPD":"#c27b31"},
                  title="Held-out classification and ranking metrics")
    fig3.update_layout(template="plotly_white", height=450, yaxis_range=[0,1], legend_title_text="")
    fig3.write_html(str(OUT/"metric_comparison.html"), include_plotlyjs="cdn")


if __name__ == "__main__":
    main()
