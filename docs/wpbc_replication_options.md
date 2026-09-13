# WPBC Replication Options

The 2021 paper reports WPBC as having 34 features, while the raw UCI file has 35 fields:

1. identifier
2. N/R outcome target
3. time
4-35. 32 numeric variables

This document records the two Phase 1 options requested. It does **not** select either option.

## Option A: 32 Numeric Variables

**Candidate input set:** raw fields 4-35, excluding identifier, outcome, and time.

**Rationale:**

- The UCI `.names` file describes 32 real-valued variables after the outcome and metadata fields.
- Identifier is metadata and should not normally be a predictive feature.
- Time is follow-up/disease-free time and may be inappropriate as an input for a recurrence classifier depending on the endpoint definition.
- This option is the cleanest semantic interpretation of “numeric variables,” but it does not reproduce the paper's stated count of 34.

**Unresolved issue:** The paper's reported 34 cannot be explained by this option alone.

## Option B: All 34 Non-Target Fields

**Candidate input set:** raw fields 1 and 3-35, excluding only the N/R outcome.

**Rationale:**

- There are exactly 34 raw fields other than the outcome, matching the paper's reported number.
- This interpretation treats identifier and time as model inputs, or at least counts them as features.
- It is numerically consistent with the paper's dataset table.

**Unresolved issue:** Identifier is explicitly an ID, and time is follow-up information. The paper does not state that either was supplied to the SVM or RL feature space, so this option risks metadata inclusion or endpoint leakage.

## Comparison

| Option | Candidate columns | Count | Supporting evidence | Risk/open question |
|---|---|---:|---|---|
| A | 32 numeric variables | 32 | UCI `.names` describes 32 real-valued variables | Does not match paper's “34 features” count |
| B | Identifier + time + 32 numeric variables | 34 | Matches the raw non-target field count and paper table numerically | May include identifier and follow-up metadata as predictors |

## Four Missing WPBC Values

The UCI documentation identifies lymph node status as missing in four cases. The local raw inspection found literal `?` markers in raw column 35 on 1-based rows 7, 29, 86, and 197.

Possible future approaches are:

1. **Documented complete-case analysis:** exclude the four affected rows, but only if the paper or a justified replication decision supports it.
2. **Documented imputation:** preserve all rows and impute the missing variable using a method chosen and recorded before evaluation. This would be a reconstruction assumption, not a paper fact.
3. **Missing-aware evaluation:** preserve all rows and use an evaluation/model path that supports missing values, if compatible with the intended Gaussian/RBF SVM protocol.
4. **Endpoint-specific investigation:** verify whether lymph node status should be treated as an input, an outcome-related field, or excluded metadata before choosing any missing-value treatment.

No approach is implemented or selected here. Raw files remain unchanged, and current preprocessing is unchanged.

## Decision Gate

Before Phase 1 RL implementation, record:

- selected option A or B and the evidence for it;
- whether time is an input, metadata, or excluded due to endpoint leakage concerns;
- whether identifier is always excluded from model inputs;
- missing-value treatment and its sensitivity alternatives;
- the resulting feature count used in the experiment log.
