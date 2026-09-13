# Replication Assumptions

This table records decisions that affect comparability with the 2021 paper. Values are intentionally left unresolved until supported by the paper, supplementary material, source code, or an explicitly documented experiment decision.

The full evidence comparison is in [replication_protocol.md](replication_protocol.md), and the concrete Phase 1 plan is in [phase1_replication_protocol.md](phase1_replication_protocol.md). Machine-readable tables are in `results/tables/`. The entries below describe the current executable baseline only; they are not claims about the original paper.

| Parameter/decision | What the paper specifies | Our implementation choice | Source/evidence | Exact or assumption |
|---|---|---|---|---|
| Train/test split | Not specified in paper | Stratified 80/20 split (`test_size=0.2`) | `experiments/baseline/run_svm_baseline.py` | Assumption |
| SVM kernel | Not specified in paper | RBF kernel (`kernel="rbf"`) | `experiments/baseline/run_svm_baseline.py` | Assumption |
| SVM hyperparameters | Not specified in paper | `C=1.0`, `gamma="scale"` | `experiments/baseline/run_svm_baseline.py` | Assumption |
| alpha | Not specified in paper | To be established | Paper review pending | Assumption |
| gamma | Not specified in paper | To be established | Paper review pending | Assumption |
| epsilon | Not specified in paper | To be established | Paper review pending | Assumption |
| Number of episodes | Not specified in paper | To be established | Paper review pending | Assumption |
| Stopping condition | Not specified in paper | To be established | Paper review pending | Assumption |
| Random seed | Not specified in paper | `random_state=42` for the split | `experiments/baseline/run_svm_baseline.py` | Assumption |
| Preprocessing | Not specified in paper | No scaling, normalization, or imputation. The current SVM sanity-check excludes four WPBC rows containing `?` only because `SVC` requires finite numeric inputs; this is provisional and does not resolve replication treatment. | Dataset inspection and baseline runner | Assumption |
| Feature encoding | Not specified in paper | Use inspected numeric feature columns as observed; no encoding or feature selection | Dataset inspection and baseline runner | Assumption |
