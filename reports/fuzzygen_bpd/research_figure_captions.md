# FuzzyGen-BPD research figure captions

All empirical panels below use the paper-defined synthetic demonstration data. They validate the implementation and illustrate the method; they are not PatchEval or Vul4J results.

**Fig. 1. FuzzyGen-BPD: Framework Architecture.** Complete workflow from vulnerable-program input and LLM patch generation through eligibility gates, behavioral-feature extraction, fuzzy BPD scoring, weight optimization, classification, ranking, and held-out evaluation.

**Fig. 2. Fuzzy Membership Profiles of Patch Disturbance Features.** Ruspini-partitioned trapezoidal membership functions used to map TestDelta, CoverageDelta, and DiffSize into low/medium/high disturbance terms.

**Fig. 3. Fuzzy Inference and Disturbance Score Transformation.** TestDelta fuzzification and the corresponding zero-order Sugeno transformation from membership degrees to a continuous fuzzy disturbance score.

**Fig. 4. Impact of Feature Weights on Candidate Patch Ranking.** BPD values for the paper’s four worked-example candidates under equal, TestDelta-focused, CoverageDelta-focused, and DiffSize-focused weight profiles.

**Fig. 5. Convergence Behaviour of Hyperparameter Optimization Algorithms.** Mean best inner-fold Macro-F1 across the equal 60-evaluation budget for Random Search, GA, TPE, GP-BO, and SMAC-style optimization.

**Fig. 6. Comparison of Optimized Feature Weights Across Algorithms.** Mean learned weights assigned to TestDelta, CoverageDelta, and DiffSize across grouped outer-fold runs.

**Fig. 7. Performance-Consistency Trade-off in Weight Optimization.** Inner-fold Macro-F1 variability and held-out Macro-F1 for mean-only and consistency-aware objectives. The consistency-aware objective uses mean F1 minus 0.5 times its inner-fold standard deviation.

**Fig. 8. Ablation Analysis of FuzzyGen-BPD Components.** Held-out Macro-F1 for all-feature, leave-one-feature-out, and single-feature fuzzy scores under program-grouped splits.

**Additional figure: Distribution of Patch Disturbance Features.** Synthetic TestDelta, CoverageDelta, and DiffSize distributions by synthetic acceptability label.

**Additional figure: Relationship Between BPD Scores and Patch Quality.** Held-out BPD score distributions by synthetic acceptability label, showing the degree of score separation in the demonstration data.

**Additional figure: Candidate Patch Ranking Across Optimization Strategies.** Candidate order produced for one held-out synthetic program by each optimization strategy.

**Additional figure: Held-Out Performance Across Vulnerability Programs.** Macro-F1 variation across repeated program-grouped held-out folds.

**Additional figure: Sensitivity of BPD Scores to Membership Function Parameters.** Local sensitivity of the synthetic score to proportional changes in the illustrative fuzzy breakpoints.

**Additional figure: Macro-F1 Performance Comparison Across Optimization Strategies.** Mean held-out Macro-F1 for the five optimizers and equal-weight BPD baseline.
