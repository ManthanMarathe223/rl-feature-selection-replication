# Replication Limitations

The original paper does not specify all operational details needed to guarantee an exact numerical reproduction.

## Unspecified Algorithm and Evaluation Details

The paper does not provide exact values or complete procedures for:

- alpha;
- gamma;
- SVM `C`;
- SVM gamma;
- train/test or cross-validation protocol;
- feature scaling;
- random seed;
- initial-state distribution;
- complete stopping protocol; and
- the precise relationship between accuracy-difference reward and AOR's state-value notation.

This project therefore labels alpha `0.5` and gamma `0.7` as reconstruction candidates from the independent FSRLearning implementation, not official paper parameters.

## WPBC Limitations

The paper reports 34 WPBC features, while the reconstructed predictive feature space uses 32 numerical variables and excludes identifier and time. This replication therefore uses the 32 numerical variables and treats the discrepancy as a limitation.

The raw WPBC file contains four missing lymph-node-status values. The current WPBC experiment uses mean imputation inside each training split through the evaluator pipeline. The raw file is unchanged, but this imputation rule is a project reconstruction assumption.

## FSRLearning Difference

The independent FSRLearning implementation supplies candidate alpha/gamma values and related state/policy mechanics, but its default reward evaluator is Random Forest with cross-validated balanced accuracy. This project uses the paper-oriented Gaussian/RBF SVM and accuracy, so FSRLearning is evidence for reconstruction candidates, not a replacement for the paper protocol.

## Interpretation Boundary

Numerical closeness to a paper result does not prove exact replication. Differences may reflect the unresolved settings above, dataset interpretation, initial-state sampling, AOR contribution interpretation, or implementation details that the paper does not report.
