from __future__ import annotations

from io import StringIO

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from src.fuzzygen_bpd import FEATURES, score_candidates, validate_candidates
from src.uploaded_figures import (
    LABEL_COLUMN,
    generate_uploaded_figures,
    validate_acceptability_labels,
)


st.set_page_config(page_title="FuzzyGen-BPD | Patch ranking", page_icon="🛡️", layout="wide")


@st.cache_data(show_spinner="Generating figures from the uploaded CSV...")
def cached_uploaded_figures(results, weights):
    return generate_uploaded_figures(results, weights)


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1, h2, h3 { font-family: 'Space Grotesk', sans-serif; letter-spacing: -0.035em; }
.hero { padding: 1.5rem 1.8rem; border-radius: 18px; background: linear-gradient(120deg,#102a43,#163e5c 55%,#087e8b); color: white; margin-bottom: 1rem; }
.hero p { color: #d6e8ef; margin-bottom: .2rem; }
.eyebrow { color: #a5e5df !important; text-transform: uppercase; letter-spacing: .12em; font-size: .75rem; font-weight: 700; }
.note { padding: 1rem 1.2rem; border-radius: 12px; background: #fff8e7; border: 1px solid #f1d99c; color: #624b15; }
.small-muted { color: #64748b; font-size: .88rem; }
</style>
""", unsafe_allow_html=True)


def demo_data():
    # Paper's Table V worked example. Gate values are illustrative inputs for this UI demo.
    return pd.DataFrame([
        ("VUL-DEMO-01", "Candidate A", 0.22, 0.035, 0.10, True, True, True),
        ("VUL-DEMO-01", "Candidate B", 0.08, 0.060, 0.38, True, True, True),
        ("VUL-DEMO-01", "Candidate C", 0.40, 0.012, 0.04, True, True, True),
        ("VUL-DEMO-01", "Candidate D", 0.55, 0.120, 0.20, True, True, True),
    ], columns=["vulnerability_id", "candidate_id", *FEATURES, "build_pass", "pov_pass", "functional_pass"])


st.markdown("""
<div class="hero">
  <p class="eyebrow">Research prototype · fuzzy decision support</p>
  <h1 style="margin:.35rem 0;color:white">FuzzyGen-BPD</h1>
  <p>Rank vulnerability patches that pass the build, proof-of-vulnerability, and functional-test gates.</p>
</div>
""", unsafe_allow_html=True)
st.markdown('<div class="note"><b>Interpretation:</b> BPD estimates behavioral disturbance; it is not a correctness or security proof. This paper is a proposal with a synthetic demonstration. Its Vul4J experiments and real-patch results are unfinished.</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("Scoring setup")
    st.caption("Feature importance weights are normalized to sum to 1.")
    raw_t = st.slider("TestDelta weight", 0.0, 1.0, 0.60, 0.01)
    raw_c = st.slider("CoverageDelta weight", 0.0, 1.0, 0.20, 0.01)
    raw_d = st.slider("DiffSize weight", 0.0, 1.0, 0.20, 0.01)
    threshold = st.slider("Decision threshold τ", 0.10, 0.90, 0.30, 0.01)
    weights_sum = raw_t + raw_c + raw_d
    if weights_sum:
        st.caption(f"Normalized: T {raw_t/weights_sum:.2f} · C {raw_c/weights_sum:.2f} · D {raw_d/weights_sum:.2f}")
    else:
        st.error("Set at least one weight above zero.")

st.subheader("Candidate data")
source = st.radio("Choose input", ["Paper example", "Upload CSV"], horizontal=True, label_visibility="collapsed")
if source == "Paper example":
    data = demo_data()
    st.caption("Four feature vectors from the paper’s worked example. Gate statuses below are demonstration values.")
else:
    uploaded = st.file_uploader("Upload candidate metrics CSV", type=["csv"])
    if uploaded is None:
        st.info("Upload a CSV with vulnerability IDs, candidate IDs, T/C/D features, the three gate outcomes, and a `patch_acceptable` true/false label.")
        template = pd.DataFrame(columns=["vulnerability_id", "candidate_id", *FEATURES, "build_pass", "pov_pass", "functional_pass", LABEL_COLUMN])
        st.download_button("Download CSV template", template.to_csv(index=False), "fuzzygen_bpd_template.csv", "text/csv")
        st.stop()
    try:
        data = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"Could not read CSV: {exc}")
        st.stop()
    if LABEL_COLUMN not in data.columns:
        data[LABEL_COLUMN] = float("nan")

try:
    data = validate_candidates(data)
except ValueError as exc:
    if "cve_id" in data.columns and "vulnerability_id" not in data.columns:
        st.warning("This looks like the PatchEval vulnerability manifest. It contains CVE metadata, not candidate-patch test measurements. For a working example, download the sample candidate CSV below. To score real patches, first generate candidate patches and measure TestDelta, CoverageDelta, DiffSize, and the three gate outcomes.")
        sample_path = "data/fuzzygen_bpd_example_candidates.csv"
        try:
            with open(sample_path, "rb") as sample_file:
                st.download_button("Download compatible example CSV", sample_file, "fuzzygen_bpd_example_candidates.csv", "text/csv")
        except OSError:
            pass
        st.stop()
    st.error(str(exc))
    st.stop()

with st.expander(
    "Review / edit input values",
    expanded=source == "Paper example"
    or (source == "Upload CSV" and data[LABEL_COLUMN].isna().any()),
):
    edited = st.data_editor(data, use_container_width=True, hide_index=True, num_rows="dynamic", key="candidate_editor")
try:
    data = validate_candidates(edited)
except ValueError as exc:
    st.error(str(exc))
    st.stop()
upload_label_error = None
if source == "Upload CSV":
    try:
        data = validate_acceptability_labels(data)
    except ValueError as exc:
        upload_label_error = str(exc)

if weights_sum == 0:
    st.stop()
results = score_candidates(data, (raw_t, raw_c, raw_d), threshold)
eligible = results[results["eligible"]]
ranked = eligible.groupby("vulnerability_id").size()
ineligible_count = int((~results["eligible"]).sum())

st.subheader("Overview")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Candidates", len(results))
c2.metric("Eligible", len(eligible))
c3.metric("Blocked by gates", ineligible_count)
c4.metric("Vulnerabilities", results["vulnerability_id"].nunique())

left, right = st.columns([1.25, 1])
with left:
    st.subheader("Ranked candidates")
    display_cols = ["vulnerability_id", "candidate_id", "eligible", "rank", *FEATURES, "F_TestDelta", "F_CoverageDelta", "F_DiffSize", "BPD", "decision"]
    if source == "Upload CSV":
        display_cols.append(LABEL_COLUMN)
    st.dataframe(results[display_cols], use_container_width=True, hide_index=True,
                 column_config={"BPD": st.column_config.NumberColumn(format="%.3f"),
                                "TestDelta": st.column_config.NumberColumn(format="%.3f"),
                                "CoverageDelta": st.column_config.NumberColumn(format="%.3f"),
                                "DiffSize": st.column_config.NumberColumn(format="%.3f"),
                                "rank": st.column_config.NumberColumn("Rank", format="%d")})
    st.download_button("Download scored results", results.to_csv(index=False), "fuzzygen_bpd_scored.csv", "text/csv")
with right:
    st.subheader("BPD comparison")
    if len(eligible):
        vulnerability_ids = sorted(eligible["vulnerability_id"].astype(str).unique())
        selected_vulnerability = st.selectbox("Vulnerability", vulnerability_ids, key="chart_vulnerability")
        chart_data = eligible[eligible["vulnerability_id"].astype(str) == selected_vulnerability]
        chart = px.bar(chart_data, x="candidate_id", y="BPD", text_auto=".3f",
                       color_discrete_sequence=["#07838d"])
        chart.add_hline(y=threshold, line_dash="dash", line_color="#d1684b", annotation_text=f"τ = {threshold:.2f}")
        chart.update_layout(showlegend=False, height=360, margin=dict(t=35, b=10))
        chart.update_yaxes(range=[0, 1], title="Behavioral Patch Distance")
        st.plotly_chart(chart, use_container_width=True)
    else:
        st.info("No candidates passed all three gates, so no ranking is produced.")

st.subheader("How the score is calculated")
st.markdown("Each feature is mapped through the paper’s Low/Medium/High (or Small/Medium/Large) trapezoidal membership functions and Sugeno singletons **0.1, 0.5, 0.9**. The weighted combination is:")
st.latex(r"\mathrm{BPD}=w_TF(T)+w_CF(C)+w_DF(D),\quad w_T+w_C+w_D=1")
st.markdown("Eligible candidates are ranked by ascending BPD within each vulnerability. A score below τ is labeled **lower disturbance**. Candidates failing any gate stay ineligible regardless of score.")
with st.expander("Feature definitions and paper breakpoints"):
    st.markdown("- **TestDelta (T):** fraction of legitimate regression-test outcomes that changed. A change signals disturbance; it is not automatically a regression.\n- **CoverageDelta (C):** absolute difference in line coverage over modified classes.\n- **DiffSize (D):** added and removed non-comment lines divided by original file lines, capped at 1.\n\nThe app uses the paper’s illustrative breakpoints. The paper proposes calibrating breakpoints on training folds; it does not provide fitted Vul4J breakpoints.")
    st.dataframe(pd.DataFrame([
        ("TestDelta", "Low", "0, 0, 0.15, 0.30"), ("TestDelta", "Medium", "0.15, 0.30, 0.50, 0.65"), ("TestDelta", "High", "0.50, 0.65, 1, 1"),
        ("CoverageDelta", "Low", "0, 0, 0.02, 0.05"), ("CoverageDelta", "Medium", "0.02, 0.05, 0.10, 0.15"), ("CoverageDelta", "High", "0.10, 0.15, 1, 1"),
        ("DiffSize", "Small", "0, 0, 0.05, 0.15"), ("DiffSize", "Medium", "0.05, 0.15, 0.30, 0.45"), ("DiffSize", "Large", "0.30, 0.45, 1, 1"),
    ], columns=["Feature", "Term", "(a, b, c, d)"]), hide_index=True, use_container_width=True)

st.caption("FuzzyGen-BPD prototype based on the provided conference-paper draft. Build/test outcomes and T/C/D measurements must be produced by external patch-analysis tooling; this app does not execute candidate code or verify security.")

st.divider()
st.subheader("Executed pipeline: synthetic evaluation")
metrics_path = "reports/fuzzygen_bpd/metrics.json"
predictions_path = "reports/fuzzygen_bpd/synthetic_heldout_predictions.csv"
if not __import__("os").path.exists(metrics_path) or not __import__("os").path.exists(predictions_path):
    st.info("Run `python scripts/run_fuzzygen_bpd_pipeline.py` to generate the paper's synthetic, program-grouped evaluation and charts.")
else:
    import json
    with open(metrics_path, encoding="utf-8") as metrics_file:
        evaluation = json.load(metrics_file)
    heldout = pd.read_csv(predictions_path)
    st.warning("Synthetic demonstration only. These metrics verify the pipeline mechanics and are not PatchEval or Vul4J results.")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Candidates / programs", f"{evaluation['candidate_count']} / {evaluation['program_count']}")
    m2.metric("Held-out Macro-F1", f"{evaluation['optimized']['macro_f1']:.3f}", help="95% program-cluster bootstrap interval: " + "–".join(f"{x:.3f}" for x in evaluation["cluster_bootstrap_95_ci"]["macro_f1"]))
    m3.metric("MCC", f"{evaluation['optimized']['mcc']:.3f}")
    m4.metric("Top-1 ranking accuracy", f"{evaluation['top1_accuracy_optimized']:.3f}", help="95% program-cluster bootstrap interval: " + "–".join(f"{x:.3f}" for x in evaluation["cluster_bootstrap_95_ci"]["top1"]))
    eval_left, eval_right = st.columns(2)
    with eval_left:
        st.markdown("**Held-out confusion matrix** · rows = actual, columns = predicted")
        cm = evaluation["optimized"]["confusion_matrix"]
        heat = go.Figure(go.Heatmap(z=cm, x=["Predicted not acceptable", "Predicted acceptable"],
                                    y=["Actual not acceptable", "Actual acceptable"],
                                    colorscale=[[0, "#eff6ff"], [1, "#145b8c"]], text=cm, texttemplate="%{text}",
                                    showscale=False, hovertemplate="%{y}<br>%{x}<br>Count: %{z}<extra></extra>"))
        heat.update_layout(template="plotly_white", height=380, margin=dict(l=15,r=15,t=20,b=75), xaxis_title="", yaxis_title="")
        st.plotly_chart(heat, use_container_width=True)
    with eval_right:
        st.markdown("**Optimized weights vs equal-weight baseline**")
        comparison = pd.DataFrame([
            {"Method":"Optimized fuzzy BPD", "Macro-F1":evaluation["optimized"]["macro_f1"], "MCC":evaluation["optimized"]["mcc"], "Top-1":evaluation["top1_accuracy_optimized"]},
            {"Method":"Equal-weight fuzzy BPD", "Macro-F1":evaluation["equal_weight_baseline"]["macro_f1"], "MCC":evaluation["equal_weight_baseline"]["mcc"], "Top-1":evaluation["top1_accuracy_equal_weight"]},
        ]).melt("Method", var_name="Metric", value_name="Score")
        bars = px.bar(comparison, x="Metric", y="Score", color="Method", barmode="group", text_auto=".3f",
                      color_discrete_map={"Optimized fuzzy BPD":"#176b87", "Equal-weight fuzzy BPD":"#c27b31"})
        bars.update_layout(template="plotly_white", height=380, yaxis_range=[0,1], margin=dict(l=15,r=15,t=20,b=20), legend_title_text="")
        st.plotly_chart(bars, use_container_width=True)
    score_plot = px.box(heldout, x="acceptable", y="bpd_optimized", points="all", color="acceptable",
                        color_discrete_map={0:"#c27b31",1:"#176b87"},
                        labels={"acceptable":"Synthetic acceptability label (0 = no, 1 = yes)","bpd_optimized":"Held-out BPD score"},
                        title="Held-out BPD scores by synthetic label")
    score_plot.update_layout(template="plotly_white", height=420, showlegend=False)
    st.plotly_chart(score_plot, use_container_width=True)
    paired = evaluation["paired_program_comparison_optimized_minus_equal"]["macro_f1"]
    st.markdown(f"**Statistical comparison:** optimized minus equal-weight Macro-F1 = {paired['mean_delta']:+.3f}; 95% paired program-cluster bootstrap CI [{paired['cluster_bootstrap_95_ci'][0]:+.3f}, {paired['cluster_bootstrap_95_ci'][1]:+.3f}], paired program-level sign-flip p = {paired['paired_sign_flip_p']:.3f}. This synthetic run does not show a reliable advantage for learned weights.")
    ranking_summary = pd.DataFrame([
        {"Method":"Optimized fuzzy BPD", **evaluation["ranking_optimized"]},
        {"Method":"Equal-weight fuzzy BPD", **evaluation["ranking_equal_weight"]},
    ])
    st.markdown("**Ranking metrics**")
    st.dataframe(ranking_summary, use_container_width=True, hide_index=True,
                 column_config={c: st.column_config.NumberColumn(format="%.3f") for c in ["top1", "mrr", "ndcg"]})
    st.download_button("Download held-out predictions", heldout.to_csv(index=False), "fuzzygen_bpd_synthetic_predictions.csv", "text/csv")

optimizer_metrics_path = "reports/fuzzygen_bpd/optimizer_comparison.json"
if __import__("os").path.exists(optimizer_metrics_path):
    with open(optimizer_metrics_path, encoding="utf-8") as optimizer_file:
        optimizer_eval = json.load(optimizer_file)
    st.divider()
    st.subheader("Competitive analysis: five optimizers")
    st.warning("Same synthetic dataset, 60 evaluations per optimizer per inner search, five program-grouped outer folds repeated across three split seeds. These are demonstration results only, not PatchEval or Vul4J findings.")
    optimizer_rows = []
    for method, values in optimizer_eval["classification_and_ranking"].items():
        f1_ci=optimizer_eval["cluster_bootstrap_95_ci"][method]["macro_f1"]
        optimizer_rows.append({"Optimizer":method,"Macro-F1":values["macro_f1"],"Macro-F1 95% CI":f"[{f1_ci[0]:.3f}, {f1_ci[1]:.3f}]","MCC":values["mcc"],"Top-1":values["top1"],"MRR":values["mrr"],"NDCG":values["ndcg"]})
    optimizer_table = pd.DataFrame(optimizer_rows)
    st.dataframe(optimizer_table, use_container_width=True, hide_index=True,
                 column_config={c: st.column_config.NumberColumn(format="%.3f") for c in ["Macro-F1","MCC","Top-1","MRR","NDCG"]})
    comp_left, comp_right = st.columns(2)
    with comp_left:
        opt_chart = px.bar(optimizer_table.melt(id_vars="Optimizer",value_vars=["Macro-F1","MCC","Top-1","MRR","NDCG"],var_name="Metric",value_name="Score"),
                           x="Optimizer",y="Score",color="Metric",barmode="group",text_auto=".2f",
                           color_discrete_map={"Macro-F1":"#176b87","MCC":"#c27b31","Top-1":"#65499b","MRR":"#2f855a","NDCG":"#cf5a4a"})
        opt_chart.update_layout(template="plotly_white",height=430,yaxis_range=[0,1],legend_title_text="",margin=dict(t=25,b=75))
        st.plotly_chart(opt_chart,use_container_width=True)
    with comp_right:
        methods=list(optimizer_eval["classification_and_ranking"])
        selected_method=st.selectbox("Confusion matrix optimizer",methods,key="optimizer_cm_method")
        cm=optimizer_eval["classification_and_ranking"][selected_method]["confusion_matrix"]
        cm_chart=go.Figure(go.Heatmap(z=cm,x=["Predicted unacceptable","Predicted acceptable"],y=["Actual unacceptable","Actual acceptable"],
                                      colorscale=[[0,"#eff6ff"],[1,"#145b8c"]],text=cm,texttemplate="%{text}",showscale=False))
        cm_chart.update_layout(template="plotly_white",height=430,margin=dict(l=10,r=10,t=25,b=75),xaxis_title="Prediction",yaxis_title="Ground truth")
        st.plotly_chart(cm_chart,use_container_width=True)
    fold_path="reports/fuzzygen_bpd/optimizer_fold_metrics.csv"
    trace_path="reports/fuzzygen_bpd/optimizer_search_traces.csv"
    if __import__("os").path.exists(fold_path):
        fold_results=pd.read_csv(fold_path)
        fold_chart=px.box(fold_results,x="method",y="macro_f1",color="method",points="all",
                          color_discrete_map={"Random Search":"#176b87","GA":"#c27b31","TPE":"#65499b","GP-BO":"#2f855a","SMAC":"#cf5a4a","Equal-weight BPD":"#607d8b"},
                          title="Held-out Macro-F1 across program-grouped folds")
        fold_chart.update_layout(template="plotly_white",showlegend=False,height=420,xaxis_title="Optimizer",yaxis_title="Macro-F1",margin=dict(t=65,b=75))
        st.plotly_chart(fold_chart,use_container_width=True)
    pairwise_path="reports/fuzzygen_bpd/optimizer_pairwise_tests.csv"
    if __import__("os").path.exists(pairwise_path):
        pairwise=pd.read_csv(pairwise_path)
        st.markdown("**Paired program-level Macro-F1 tests** · Wilcoxon signed-rank with Holm correction")
        st.dataframe(pairwise,use_container_width=True,hide_index=True,
                     column_config={c:st.column_config.NumberColumn(format="%.3f") for c in ["mean_macro_f1_delta","ci_low","ci_high","wilcoxon_p","holm_adjusted_p"]})
    st.caption(optimizer_eval["method_note"])

figure_manifest_path = "reports/fuzzygen_bpd/research_figure_manifest.json"
st.divider()
st.subheader("Research figure collection")
figure_set = st.radio(
    "Figure set",
    ["Paper example", "Uploaded CSV"],
    index=1 if source == "Upload CSV" else 0,
    horizontal=True,
    key="figure_set",
)

if figure_set == "Uploaded CSV":
    if source != "Upload CSV":
        st.info("Choose “Upload CSV” under Candidate data to generate figures from uploaded candidates.")
    elif upload_label_error:
        st.error(upload_label_error)
    else:
        st.warning(
            "Uploaded-data optimizer figures use `patch_acceptable` labels and grouped "
            "cross-validation when multiple vulnerabilities are present. They are exploratory "
            "tuning results on this upload, not independent test-set findings."
        )
        try:
            figures = cached_uploaded_figures(results, (raw_t, raw_c, raw_d))
        except ValueError as exc:
            st.error(str(exc))
        else:
            selected_figure = st.selectbox(
                "Choose a figure", list(figures), key="uploaded_research_figure"
            )
            description, figure, height = figures[selected_figure]
            st.caption(description)
            figure.update_layout(height=height)
            st.plotly_chart(
                figure,
                use_container_width=True,
                config={"responsive": True, "displaylogo": False},
            )
else:
    if not __import__("os").path.exists(figure_manifest_path):
        st.info(
            "Paper-example figures are not available. Run "
            "`python scripts/generate_fuzzygen_research_figures.py` to generate them."
        )
    else:
        with open(figure_manifest_path, encoding="utf-8") as figure_manifest_file:
            figure_manifest = json.load(figure_manifest_file)
        st.warning(
            "Paper-example charts use the paper-defined synthetic demonstration data. "
            "They must not be described as PatchEval or Vul4J findings."
        )
        architecture_path = (
            "/Users/ankitkumarsingh/Desktop/FuzzyGen-BPD Research Framework Infographic.png"
        )
        figures = {
            "Fig. 1 · FuzzyGen-BPD: Framework Architecture": (
                architecture_path,
                "Complete workflow of the proposed system",
                950,
            ),
            "Fig. 2 · Fuzzy Membership Profiles of Patch Disturbance Features": (
                "reports/fuzzygen_bpd/fig_02_membership_profiles.html",
                "Low, Medium, and High membership curves",
                500,
            ),
            "Fig. 3 · Fuzzy Inference and Disturbance Score Transformation": (
                "reports/fuzzygen_bpd/fig_03_inference_transformation.html",
                "Input fuzzification and Sugeno crisp-output transformation",
                500,
            ),
            "Fig. 4 · Impact of Feature Weights on Candidate Patch Ranking": (
                "reports/fuzzygen_bpd/fig_04_weight_impact.html",
                "How learned feature emphasis changes candidate ordering",
                540,
            ),
            "Fig. 5 · Convergence Behaviour of Hyperparameter Optimization Algorithms": (
                "reports/fuzzygen_bpd/fig_05_optimizer_convergence.html",
                "GA, TPE, GP-BO, SMAC, and Random Search under an equal budget",
                540,
            ),
            "Fig. 6 · Comparison of Optimized Feature Weights Across Algorithms": (
                "reports/fuzzygen_bpd/fig_06_learned_weights.html",
                "Average learned weights for TestDelta, CoverageDelta, and DiffSize",
                540,
            ),
            "Fig. 7 · Performance–Consistency Trade-off in Weight Optimization": (
                "reports/fuzzygen_bpd/fig_07_performance_consistency.html",
                "Mean-only versus consistency-aware threshold selection",
                540,
            ),
            "Fig. 8 · Ablation Analysis of FuzzyGen-BPD Components": (
                "reports/fuzzygen_bpd/fig_08_ablation.html",
                "Contribution of the three fuzzy disturbance features",
                560,
            ),
            "Additional · Distribution of Patch Disturbance Features": (
                "reports/fuzzygen_bpd/additional_01_feature_distributions.html",
                "Synthetic distribution of TestDelta, CoverageDelta, and DiffSize",
                500,
            ),
            "Additional · Relationship Between BPD Scores and Patch Quality": (
                "reports/fuzzygen_bpd/additional_02_bpd_quality.html",
                "Held-out BPD score distribution by synthetic label",
                500,
            ),
            "Additional · Candidate Patch Ranking Across Optimization Strategies": (
                "reports/fuzzygen_bpd/additional_03_ranking_strategies.html",
                "A candidate set ranked by each search method",
                520,
            ),
            "Additional · Held-Out Performance Across Vulnerability Programs": (
                "reports/fuzzygen_bpd/additional_04_heldout_generalization.html",
                "Macro-F1 variation across grouped held-out folds",
                520,
            ),
            "Additional · Sensitivity of BPD Scores to Membership Function Parameters": (
                "reports/fuzzygen_bpd/additional_05_membership_sensitivity.html",
                "Local robustness diagnostic for fuzzy membership breakpoints",
                500,
            ),
            "Additional · Macro-F1 Performance Comparison Across Optimization Strategies": (
                "reports/fuzzygen_bpd/additional_06_macro_f1_comparison.html",
                "Mean grouped-fold Macro-F1 by method",
                500,
            ),
        }
        selected_figure = st.selectbox(
            "Choose a figure", list(figures), key="paper_research_figure"
        )
        selected_path, selected_description, selected_height = figures[selected_figure]
        st.caption(selected_description)
        if selected_path.lower().endswith(".png"):
            if __import__("os").path.isfile(selected_path):
                st.image(selected_path, use_container_width=True)
            else:
                st.info("The framework architecture image is not included in this repository.")
        else:
            try:
                with open(selected_path, encoding="utf-8") as selected_file:
                    figure_html = selected_file.read()
                components.html(figure_html, height=selected_height, scrolling=True)
                st.download_button(
                    "Download this interactive figure",
                    figure_html,
                    __import__("os").path.basename(selected_path),
                    "text/html",
                )
            except OSError:
                st.info(
                    "This figure file is not available. Run "
                    "`python scripts/generate_fuzzygen_research_figures.py` to regenerate it."
                )
