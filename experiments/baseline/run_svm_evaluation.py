"""Run the configurable Phase 1A SVM evaluation baseline."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Literal

import numpy as np

from src.data.inspect_datasets import Inspection, PROJECT_ROOT, dataset_specs, load_dataset
from src.evaluation.evaluation_config import EvaluationConfig
from src.evaluation.svm_evaluator import SVMEvaluator


OUTPUT_PATH = PROJECT_ROOT / "results" / "tables" / "svm_evaluation_baseline.csv"
FeaturePolicy = Literal["numeric_only", "all_non_target"]


def _feature_columns(
    inspection: Inspection,
    policy: FeaturePolicy = "numeric_only",
    *,
    include_identifier: bool = False,
) -> tuple[str, ...]:
    """Resolve feature columns without silently including WPBC metadata."""
    if policy == "numeric_only":
        return inspection.spec.feature_columns
    if policy != "all_non_target":
        raise ValueError("Unsupported feature policy.")
    columns = tuple(
        column
        for column in inspection.frame.columns
        if column != inspection.spec.target_column
        and (include_identifier or column != "identifier")
    )
    return columns


def _prepare_arrays(
    inspection: Inspection,
    feature_columns: tuple[str, ...],
    missing_value_policy: str,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Apply only the explicitly selected missing-value policy."""
    frame = inspection.frame.loc[:, list(feature_columns)].apply(
        lambda column: column if np.issubdtype(column.dtype, np.number)
        else column.astype(float)
    )
    targets = inspection.frame.loc[:, inspection.spec.target_column]
    missing_rows = frame.isna().any(axis=1) | targets.isna()
    missing_count = int(missing_rows.sum())
    if missing_count and missing_value_policy == "error":
        raise ValueError(
            f"{inspection.spec.name} contains {missing_count} rows with missing values; "
            "choose an explicit missing_value_policy."
        )
    if missing_count and missing_value_policy == "drop_rows":
        frame = frame.loc[~missing_rows]
        targets = targets.loc[~missing_rows]
    return frame.to_numpy(dtype=float), targets.to_numpy(), missing_count


def run_evaluation(
    config: EvaluationConfig | None = None,
    *,
    feature_policy: FeaturePolicy = "numeric_only",
    include_identifier: bool = False,
) -> list[dict[str, object]]:
    """Evaluate all datasets with all features permitted by the policy."""
    config = config or EvaluationConfig(missing_value_policy="drop_rows")
    records: list[dict[str, object]] = []
    for spec in dataset_specs():
        inspection = load_dataset(spec)
        columns = _feature_columns(
            inspection,
            feature_policy,
            include_identifier=include_identifier,
        )
        features, targets, missing_count = _prepare_arrays(
            inspection,
            columns,
            config.missing_value_policy,
        )
        result = SVMEvaluator(config).evaluate(
            features,
            targets,
            selected_features=tuple(range(features.shape[1])),
        )
        records.append(
            {
                "dataset": spec.name,
                "feature_policy": feature_policy,
                "include_identifier": include_identifier,
                "number_of_features": result.number_of_selected_features,
                "number_of_samples": result.number_of_samples,
                "missing_rows_observed": missing_count,
                "accuracy": result.accuracy,
                "runtime_seconds": result.runtime_seconds,
                "training_time_seconds": result.training_time_seconds,
                "evaluation_strategy": result.evaluation_strategy,
                "model_configuration": json.dumps(result.model_configuration, sort_keys=True),
            }
        )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return records


def main() -> None:
    """Run the explicit numeric-only baseline and print its summary."""
    config = EvaluationConfig(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        scaling="none",
        random_state=42,
        evaluation_strategy="holdout",
        scoring_metric="accuracy",
        test_size=0.2,
        missing_value_policy="drop_rows",
    )
    for record in run_evaluation(config):
        print(f"dataset: {record['dataset']}")
        print(f"number of features: {record['number_of_features']}")
        print(f"number of samples: {record['number_of_samples']}")
        print(f"accuracy: {float(record['accuracy']):.6f}")
        print(f"runtime: {float(record['runtime_seconds']):.6f} seconds")
    print(f"Saved evaluation results: {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
