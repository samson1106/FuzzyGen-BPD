"""Equal-budget optimizer comparison for the synthetic FuzzyGen-BPD protocol.

Methods: Random Search, Genetic Algorithm, TPE, GP expected improvement, and
random-forest SMAC-style expected improvement. Uses grouped nested CV on the
paper's synthetic demonstration only; it is not PatchEval or Vul4J evidence.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.special import ndtr
from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from run_fuzzygen_bpd_pipeline import (ROOT, OUT, FOLDS, RNG_SEED, synthesize,
    groups_to_folds, bpd_matrix, macro_f1, classification_metrics, ranking_metrics,
    cluster_bootstrap, paired_program_test)

METHODS = ("Random Search", "GA", "TPE", "GP-BO", "SMAC")
BASELINE = "Equal-weight BPD"
COMPARISON_METHODS = METHODS + (BASELINE,)
BUDGET = 60


def to_weights(theta):
    raw = np.clip(np.asarray(theta[:3], dtype=float), 1e-8, 1.)
    return raw / raw.sum()


def evaluate_optimizer(name, objective, seed, budget=BUDGET):
    """Return best parameters and best-so-far values under an equal budget."""
    rng = np.random.default_rng(seed)
    X, y = [], []
    trace = []
    def ask_random():
        return rng.random(4)
    def evaluate(x):
        X.append(np.asarray(x, dtype=float))
        y.append(float(objective(X[-1])))
        trace.append(max(y))
    if name == "Random Search":
        for _ in range(budget): evaluate(ask_random())
    elif name == "GA":
        population = [ask_random() for _ in range(12)]
        for x in population: evaluate(x)
        while len(y) < budget:
            # size-2 tournament, blend crossover, Gaussian mutation and elitism.
            parents = []
            for _ in range(2):
                contenders = rng.choice(len(y), 2, replace=False)
                parents.append(X[contenders[int(np.argmax(np.asarray(y)[contenders]))]])
            alpha = rng.uniform(-.25, 1.25, size=4)
            child = np.clip(alpha*parents[0] + (1-alpha)*parents[1], 0, 1)
            mutate = rng.random(4) < .25
            child[mutate] = np.clip(child[mutate] + rng.normal(0, .1, mutate.sum()), 0, 1)
            evaluate(child)
    else:
        n_initial = 10
        for _ in range(n_initial): evaluate(ask_random())
        while len(y) < budget:
            pool = rng.random((128, 4))
            xa, ya = np.asarray(X), np.asarray(y)
            if name == "TPE":
                good_n = max(5, int(.2*len(y)))
                good_idx = np.argsort(ya)[-good_n:]
                bad_idx = np.argsort(ya)[:-good_n]
                good, bad = xa[good_idx], xa[bad_idx]
                bw_good = np.maximum(np.std(good, axis=0), .06)
                bw_bad = np.maximum(np.std(bad, axis=0), .06)
                # Independent Parzen mixtures; maximize l(theta)/g(theta).
                def log_density(candidates, centers, bandwidth):
                    z = (candidates[:, None, :] - centers[None, :, :]) / bandwidth[None, None, :]
                    terms = -.5*np.sum(z*z, axis=2) - np.log(bandwidth).sum()
                    m = terms.max(axis=1, keepdims=True)
                    return (m[:,0] + np.log(np.exp(terms-m).mean(axis=1)))
                log_ratio = log_density(pool, good, bw_good)-log_density(pool, bad, bw_bad)
                candidate = pool[int(np.argmax(log_ratio))]
            elif name == "GP-BO":
                kernel = ConstantKernel(1.0)*Matern(length_scale=np.full(4, .35), nu=2.5)+WhiteKernel(noise_level=.01)
                # Keep the kernel fixed for fast, reproducible EI updates; each
                # optimizer still receives exactly the same objective budget.
                gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, optimizer=None, random_state=seed)
                gp.fit(xa, ya)
                mean, std = gp.predict(pool, return_std=True)
                improvement = mean.max()-mean
                z = improvement/np.maximum(std, 1e-9)
                ei = improvement*ndtr(z)+std*np.exp(-.5*z*z)/np.sqrt(2*np.pi)
                candidate = pool[int(np.argmax(ei))]
            else:  # SMAC-style RF surrogate and expected improvement
                rf = RandomForestRegressor(n_estimators=48, min_samples_leaf=2, random_state=seed+n_initial+len(y), n_jobs=1)
                rf.fit(xa, ya)
                tree_preds = np.column_stack([tree.predict(pool) for tree in rf.estimators_])
                mean, std = tree_preds.mean(axis=1), tree_preds.std(axis=1)
                improvement = mean.max()-mean
                z = improvement/np.maximum(std, 1e-9)
                ei = improvement*ndtr(z)+std*np.exp(-.5*z*z)/np.sqrt(2*np.pi)
                candidate = pool[int(np.argmax(ei))]
            evaluate(candidate)
    best_idx = int(np.argmax(y))
    return X[best_idx], y[best_idx], trace


def holm_adjust(p_values):
    names = list(p_values)
    ordered = sorted(names, key=lambda k: p_values[k])
    adjusted, running = {}, 0.
    m = len(ordered)
    for rank, name in enumerate(ordered):
        running = max(running, min(1., (m-rank)*p_values[name]))
        adjusted[name] = running
    return adjusted


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = synthesize()
    prediction_rows, fold_rows, weight_rows, trace_rows = [], [], [], []
    fold_number = 0
    for split_seed in range(3):
        outer = groups_to_folds(data.program_id.to_numpy(), FOLDS, np.random.default_rng(split_seed))
        for test_idx in outer:
            fold_number += 1
            train_idx = np.setdiff1d(np.arange(len(data)), test_idx)
            train, test = data.iloc[train_idx].reset_index(drop=True), data.iloc[test_idx].copy()
            inner_indices = groups_to_folds(train.program_id.to_numpy(), 4, np.random.default_rng(500+fold_number))
            Xtrain, ytrain = bpd_matrix(train), train.acceptable.to_numpy(dtype=int)
            def objective(theta):
                weights = to_weights(theta)
                threshold = .1 + .8*float(theta[3])
                scores = Xtrain @ weights
                return float(np.mean([macro_f1(ytrain[ix], (scores[ix] < threshold).astype(int)) for ix in inner_indices]))
            for method_idx, method in enumerate(COMPARISON_METHODS):
                if method == BASELINE:
                    w = np.array([1/3]*3)
                    scores_train = Xtrain @ w
                    best_inner, threshold = -1., .5
                    for candidate_threshold in np.linspace(.1,.9,17):
                        value = float(np.mean([macro_f1(ytrain[ix],(scores_train[ix]<candidate_threshold).astype(int)) for ix in inner_indices]))
                        if value > best_inner:
                            best_inner, threshold = value, float(candidate_threshold)
                    trace = []
                else:
                    params, best_inner, trace = evaluate_optimizer(method, objective, 7000+split_seed*100+fold_number*10+method_idx)
                    w, threshold = to_weights(params), .1+.8*float(params[3])
                values = bpd_matrix(test) @ w
                preds = (values < threshold).astype(int)
                rank_col = pd.Series(values, index=test.index).groupby(test.program_id).rank(method="min").to_numpy()
                method_frame = test[["program_id","candidate_id","acceptable"]].copy()
                method_frame["seed"] = split_seed
                method_frame["outer_fold"] = fold_number
                method_frame["method"] = method
                method_frame["bpd"] = values
                method_frame["threshold"] = threshold
                method_frame["prediction"] = preds
                method_frame["rank"] = rank_col
                prediction_rows.append(method_frame)
                fold_metrics = classification_metrics(test.acceptable.to_numpy(dtype=int), preds)
                rank_frame = method_frame.rename(columns={"rank":"rank_score"}).copy()
                rank_values = ranking_metrics(rank_frame.rename(columns={"rank_score":"rank"}), "rank")
                fold_rows.append({"seed":split_seed,"fold":fold_number,"method":method,"programs":test.program_id.nunique(),"candidates":len(test),"inner_macro_f1":best_inner,**{k:v for k,v in fold_metrics.items() if k!="confusion_matrix"},**rank_values})
                weight_rows.append({"seed":split_seed,"fold":fold_number,"method":method,"w_testdelta":w[0],"w_coveragedelta":w[1],"w_diffsize":w[2],"threshold":threshold})
                trace_rows.extend({"seed":split_seed,"fold":fold_number,"method":method,"evaluation":i+1,"best_inner_macro_f1":v} for i,v in enumerate(trace))
    pred = pd.concat(prediction_rows, ignore_index=True)
    folds = pd.DataFrame(fold_rows)
    weights = pd.DataFrame(weight_rows)
    traces = pd.DataFrame(trace_rows)
    # Group ranking by program inside each held-out repetition.
    pred["rank_group"] = pred.seed.astype(str)+"_"+pred.program_id
    aggregate, confusions, cis = {}, {}, {}
    for method in COMPARISON_METHODS:
        part = pred[pred.method == method].copy()
        aggregate[method] = classification_metrics(part.acceptable.to_numpy(dtype=int), part.prediction.to_numpy(dtype=int))
        aggregate[method].update(ranking_metrics(part.rename(columns={"rank":"rank_value"}), "rank_value"))
        ci = cluster_bootstrap(part.rename(columns={"prediction":"pred_boot"}), "pred_boot", "rank", seed=777+COMPARISON_METHODS.index(method), reps=2000)
        cis[method] = ci
        confusions[method] = aggregate[method]["confusion_matrix"]
    # Paired program-level Macro-F1 comparisons, with Holm correction across ten pairs.
    per_program = []
    for program, part in pred.groupby("program_id"):
        row = {"program_id":program}
        for method in COMPARISON_METHODS:
            m = part[part.method == method]
            row[method] = macro_f1(m.acceptable.to_numpy(dtype=int), m.prediction.to_numpy(dtype=int))
        per_program.append(row)
    pg = pd.DataFrame(per_program)
    comparisons, raw_p = [], {}
    for i, a in enumerate(COMPARISON_METHODS):
        for b in COMPARISON_METHODS[i+1:]:
            delta = pg[a].to_numpy()-pg[b].to_numpy()
            boot_rng = np.random.default_rng(300+len(comparisons))
            boot = np.array([np.mean(boot_rng.choice(delta, len(delta), replace=True)) for _ in range(4000)])
            try: p = float(wilcoxon(delta, zero_method="wilcox", alternative="two-sided").pvalue)
            except ValueError: p = 1.
            key=f"{a} vs {b}"; raw_p[key]=p
            comparisons.append({"comparison":key,"mean_macro_f1_delta":float(delta.mean()),"ci_low":float(np.quantile(boot,.025)),"ci_high":float(np.quantile(boot,.975)),"wilcoxon_p":p})
    adjusted=holm_adjust(raw_p)
    for row in comparisons: row["holm_adjusted_p"] = adjusted[row["comparison"]]
    fold_summary=folds.groupby("method").agg(macro_f1_mean=("macro_f1","mean"),macro_f1_sd=("macro_f1","std"),mcc_mean=("mcc","mean"),top1_mean=("top1","mean"),mrr_mean=("mrr","mean"),ndcg_mean=("ndcg","mean")).reset_index()
    metrics={"status":"SYNTHETIC DEMONSTRATION ONLY; NOT PATCH EVAL OR VUL4J RESULTS","data_seed":RNG_SEED,"candidates":len(data),"programs":int(data.program_id.nunique()),"labels":{"unacceptable":int((data.acceptable==0).sum()),"acceptable":int((data.acceptable==1).sum())},"outer_folds":15,"inner_folds":4,"evaluations_per_optimizer_per_fold":BUDGET,"classification_and_ranking":aggregate,"cluster_bootstrap_95_ci":cis,"fold_summary":fold_summary.to_dict(orient="records"),"paired_program_wilcoxon_holm":comparisons,"weights":weights.groupby("method")[["w_testdelta","w_coveragedelta","w_diffsize","threshold"]].mean().round(4).to_dict(orient="index"),"method_note":"Random Search, GA, TPE, GP-BO and SMAC receive 60 evaluations per outer fold; equal-weight BPD is a baseline with its threshold tuned on inner folds. GP-BO uses a Gaussian-process expected-improvement surrogate; SMAC uses a random-forest expected-improvement surrogate; TPE uses independent Parzen density ratios. These are transparent lightweight implementations, not claims of bit-for-bit equivalence to Optuna, SMAC3, or scikit-optimize.","interpretation":"Synthetic data from the distributions described in the paper. Results test pipeline mechanics only; they do not estimate vulnerability-patch performance."}
    pred.to_csv(OUT/"optimizer_heldout_predictions.csv",index=False)
    folds.to_csv(OUT/"optimizer_fold_metrics.csv",index=False)
    weights.to_csv(OUT/"optimizer_weights.csv",index=False)
    traces.to_csv(OUT/"optimizer_search_traces.csv",index=False)
    pg.to_csv(OUT/"optimizer_program_scores.csv",index=False)
    pd.DataFrame(comparisons).to_csv(OUT/"optimizer_pairwise_tests.csv",index=False)
    (OUT/"optimizer_comparison.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
    make_figures(folds,traces,aggregate,confusions,cis)
    print(json.dumps(metrics,indent=2))


def make_figures(folds,traces,aggregate,confusions,cis):
    order=list(COMPARISON_METHODS)
    palette={"Random Search":"#176b87","GA":"#c27b31","TPE":"#65499b","GP-BO":"#2f855a","SMAC":"#cf5a4a",BASELINE:"#607d8b"}
    chart=px.box(folds,x="method",y="macro_f1",color="method",category_orders={"method":order},points="all",
                 color_discrete_map=palette,
                 title="Five-optimizer comparison · held-out Macro-F1 across 15 grouped folds")
    chart.update_layout(template="plotly_white",showlegend=False,height=520,xaxis_title="Optimizer",yaxis_title="Macro-F1",margin=dict(t=80,b=90))
    chart.write_html(str(OUT/"optimizer_comparison.html"),include_plotlyjs="cdn")
    conv=px.line(traces,x="evaluation",y="best_inner_macro_f1",color="method",line_group="seed",facet_col="method",facet_col_wrap=2,
                 category_orders={"method":order},color_discrete_sequence=["#176b87","#c27b31","#65499b","#2f855a","#cf5a4a"],
                 title="Search convergence under the equal 60-evaluation budget")
    conv.update_layout(template="plotly_white",height=800,showlegend=False,xaxis_title="Objective evaluations",yaxis_title="Best inner-fold Macro-F1")
    conv.write_html(str(OUT/"optimizer_convergence.html"),include_plotlyjs="cdn")
    fig=make_subplots(rows=2,cols=3,subplot_titles=COMPARISON_METHODS,horizontal_spacing=.10,vertical_spacing=.20)
    for i,method in enumerate(COMPARISON_METHODS):
        row,col=i//3+1,i%3+1
        cm=np.asarray(confusions[method])
        fig.add_trace(go.Heatmap(z=cm,x=["Pred. 0","Pred. 1"],y=["Actual 0","Actual 1"],
                                 colorscale=[[0,"#eff6ff"],[1,"#145b8c"]],zmin=0,zmax=max(np.max(x) for x in confusions.values()),
                                 text=cm,texttemplate="%{text}",showscale=False,hovertemplate="%{y}<br>%{x}: %{z}<extra></extra>"),row=row,col=col)
    fig.update_layout(title="Held-out confusion matrices · all methods · synthetic evaluation",template="plotly_white",height=720,margin=dict(t=95,b=45,l=85,r=35))
    fig.update_xaxes(title_text="Prediction")
    fig.update_yaxes(title_text="Actual class")
    fig.write_html(str(OUT/"optimizer_confusion_matrices.html"),include_plotlyjs="cdn")


if __name__=="__main__": main()
