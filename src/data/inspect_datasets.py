"""Inspect the original raw datasets without preprocessing them.

Run from the project root with::

    python -m src.data.inspect_datasets

The inspection deliberately preserves every parsed column. It does not scale,
impute, normalize, select features, or remove duplicate rows.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = PROJECT_ROOT / "data" / "raw"
TABLE_PATH = PROJECT_ROOT / "results" / "tables" / "dataset_inspection.csv"
REPORT_PATH = PROJECT_ROOT / "results" / "dataset_inspection_report.txt"


@dataclass(frozen=True)
class DatasetSpec:
    """Raw-file parsing and semantic column-role metadata."""

    name: str
    path: Path
    separator: str
    target_column: str
    feature_columns: tuple[str, ...]
    metadata_columns: tuple[str, ...]


@dataclass
class Inspection:
    """All reportable observations for one dataset."""

    spec: DatasetSpec
    frame: pd.DataFrame
    raw_missing_markers: dict[str, list[int]]


def _read_australian(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep=r"\s+", header=None, dtype=str, engine="python")
    frame.columns = [*(f"feature_{i}" for i in range(1, 15)), "target"]
    return frame.apply(pd.to_numeric, errors="coerce")


def _read_wpbc(path: Path) -> pd.DataFrame:
    columns = ["identifier", "target", "time"]
    columns.extend(f"feature_{i}" for i in range(1, 33))
    frame = pd.read_csv(path, header=None, names=columns, dtype=str, na_values=["?"])
    numeric_columns = ["time", *[f"feature_{i}" for i in range(1, 31)]]
    numeric_columns.extend(["feature_31", "feature_32"])
    frame[numeric_columns] = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
    return frame


def _read_sonar(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, header=None, dtype=str)
    frame.columns = [*(f"feature_{i}" for i in range(1, 61)), "target"]
    feature_columns = [f"feature_{i}" for i in range(1, 61)]
    frame[feature_columns] = frame[feature_columns].apply(pd.to_numeric, errors="coerce")
    return frame


def _wpbc_marker_rows(path: Path, column_number: int) -> list[int]:
    """Return 1-based raw row numbers containing a literal question mark."""
    rows: list[int] = []
    for row_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        values = line.split(",")
        if "?" in values:
            marker_columns = [index + 1 for index, value in enumerate(values) if value == "?"]
            if marker_columns != [column_number]:
                raise ValueError(
                    f"Unexpected WPBC missing-marker columns on raw row {row_number}: "
                    f"{marker_columns}"
                )
            rows.append(row_number)
    return rows


def dataset_specs() -> tuple[DatasetSpec, ...]:
    """Return the three explicitly defined raw dataset schemas."""
    return (
        DatasetSpec(
            "Australian Credit Approval",
            RAW_ROOT / "australian" / "australian.dat",
            r"\s+",
            "target",
            tuple(f"feature_{i}" for i in range(1, 15)),
            (),
        ),
        DatasetSpec(
            "Wisconsin Breast Cancer Prognostic (WPBC)",
            RAW_ROOT / "wpbc" / "wpbc.data",
            ",",
            "target",
            tuple(f"feature_{i}" for i in range(1, 33)),
            ("identifier", "time"),
        ),
        DatasetSpec(
            "Connectionist Bench Sonar",
            RAW_ROOT / "sonar" / "sonar.all-data",
            ",",
            "target",
            tuple(f"feature_{i}" for i in range(1, 61)),
            (),
        ),
    )


def load_dataset(spec: DatasetSpec) -> Inspection:
    """Load one raw dataset while retaining all rows and columns."""
    if not spec.path.exists():
        raise FileNotFoundError(f"Raw dataset not found: {spec.path}")
    if spec.name.startswith("Australian"):
        frame = _read_australian(spec.path)
    elif spec.name.startswith("Wisconsin"):
        frame = _read_wpbc(spec.path)
    else:
        frame = _read_sonar(spec.path)

    marker_rows = _wpbc_marker_rows(spec.path, 35) if "WPBC" in spec.name else []
    return Inspection(spec, frame, {"feature_32": marker_rows} if marker_rows else {})


def _display_value(value: Any) -> str:
    if pd.isna(value):
        return "missing"
    return str(value)


def _inspection_rows(inspection: Inspection) -> list[dict[str, str]]:
    spec, frame = inspection.spec, inspection.frame
    rows: list[dict[str, str]] = []

    def add(section: str, item: str, value: Any, details: str = "") -> None:
        rows.append(
            {
                "dataset": spec.name,
                "section": section,
                "item": item,
                "value": _display_value(value),
                "details": details,
            }
        )

    add("dataset", "rows", len(frame))
    add("dataset", "columns", len(frame.columns))
    add("dataset", "feature_count", len(spec.feature_columns))
    add("dataset", "target_column", spec.target_column)
    add("dataset", "metadata_columns", ", ".join(spec.metadata_columns) or "None")
    add("dataset", "duplicate_rows", int(frame.duplicated().sum()))
    add("dataset", "missing_values_total", int(frame.isna().sum().sum()))

    for column in frame.columns:
        role = "target" if column == spec.target_column else (
            "metadata" if column in spec.metadata_columns else "feature"
        )
        add("column_role", column, role)
        add("data_type", column, str(frame[column].dtype))
        add("missing_values", column, int(frame[column].isna().sum()))

        if pd.api.types.is_numeric_dtype(frame[column]):
            add("numeric_range", column, frame[column].min(), f"role={role}; minimum")
            add("numeric_range", column, frame[column].max(), f"role={role}; maximum")

    for label, count in frame[spec.target_column].value_counts(dropna=False).items():
        add("class_distribution", _display_value(label), int(count), "target values")

    for column, row_numbers in inspection.raw_missing_markers.items():
        add(
            "raw_missing_marker",
            column,
            len(row_numbers),
            f"literal '?' found on 1-based raw rows: {', '.join(map(str, row_numbers))}",
        )
    return rows


def _format_distribution(frame: pd.DataFrame, target: str) -> str:
    return ", ".join(
        f"{_display_value(label)}={count}"
        for label, count in frame[target].value_counts(dropna=False).items()
    )


def _render_report(inspections: list[Inspection]) -> str:
    blocks: list[str] = [
        "DATASET INSPECTION REPORT",
        "Generated by src.data.inspect_datasets.py",
        "No scaling, normalization, imputation, feature selection, or row/column dropping was performed.",
        "",
    ]
    for inspection in inspections:
        spec, frame = inspection.spec, inspection.frame
        blocks.extend(
            [
                f"=== {spec.name} ===",
                f"Raw file: {spec.path.relative_to(PROJECT_ROOT)}",
                f"Rows: {len(frame)}",
                f"Columns: {len(frame.columns)}",
                f"Feature count: {len(spec.feature_columns)}",
                f"Input features: {', '.join(spec.feature_columns)}",
                f"Target variable: {spec.target_column}",
                f"Identifier/metadata columns: {', '.join(spec.metadata_columns) or 'None'}",
                f"Duplicate rows: {int(frame.duplicated().sum())}",
                f"Missing values (parsed): {int(frame.isna().sum().sum())}",
                f"Class distribution: {_format_distribution(frame, spec.target_column)}",
                "",
                "Data types:",
                *[f"  {column}: {frame[column].dtype}" for column in frame.columns],
                "",
                "Missing values by column:",
                *[f"  {column}: {int(frame[column].isna().sum())}" for column in frame.columns],
                "",
                "Numeric minimum/maximum:",
            ]
        )
        for column in frame.columns:
            if pd.api.types.is_numeric_dtype(frame[column]):
                blocks.append(f"  {column}: min={frame[column].min()}, max={frame[column].max()}")

        if inspection.raw_missing_markers:
            blocks.extend(
                [
                    "",
                    "WPBC raw missing-value markers:",
                    "  Literal '?' markers were found in raw column 35 (feature_32) on "
                    + ", ".join(
                        f"1-based raw row {row}"
                        for row in inspection.raw_missing_markers["feature_32"]
                    )
                    + ".",
                    "  These markers are reported as missing and were not imputed or removed.",
                    "  The raw WPBC layout contains identifier, N/R target, time, and 32 numeric "
                    "fields. The paper reports 34 WPBC features, which is inconsistent with the "
                    "semantic input-feature count after separating identifier and metadata. This "
                    "inspection records both facts and does not resolve the paper's preprocessing protocol.",
                ]
            )
        blocks.extend(
            [
                "",
                "Paper/dataset consistency flag:",
                "  Dataset structure is reported as observed. Any preprocessing or target interpretation "
                "needed for paper replication remains unresolved and is not applied here.",
                "",
            ]
        )
    return "\n".join(blocks)


def inspect_datasets() -> list[Inspection]:
    """Inspect all configured datasets and save CSV and text reports."""
    inspections = [load_dataset(spec) for spec in dataset_specs()]
    rows = [row for inspection in inspections for row in _inspection_rows(inspection)]
    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TABLE_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", "section", "item", "value", "details"])
        writer.writeheader()
        writer.writerows(rows)
    REPORT_PATH.write_text(_render_report(inspections), encoding="utf-8")
    return inspections


def _print_summary(inspections: list[Inspection]) -> None:
    for inspection in inspections:
        spec, frame = inspection.spec, inspection.frame
        print(f"\n=== {spec.name} ===")
        print(f"Rows: {len(frame)} | Columns: {len(frame.columns)} | Features: {len(spec.feature_columns)}")
        print(f"Features: {', '.join(spec.feature_columns)}")
        print(f"Target: {spec.target_column} | Metadata: {', '.join(spec.metadata_columns) or 'None'}")
        print(f"Missing values: {int(frame.isna().sum().sum())} | Duplicate rows: {int(frame.duplicated().sum())}")
        print(f"Class distribution: {_format_distribution(frame, spec.target_column)}")
        if inspection.raw_missing_markers:
            print(
                "WPBC literal '?' markers: raw column 35 (feature_32), "
                + ", ".join(f"raw row {row}" for row in inspection.raw_missing_markers["feature_32"])
            )
        print("Data types and numeric ranges:")
        for column in frame.columns:
            details = f"dtype={frame[column].dtype}"
            if pd.api.types.is_numeric_dtype(frame[column]):
                details += f", min={frame[column].min()}, max={frame[column].max()}"
            print(f"  {column}: {details}")
    print(f"\nSaved summary table: {TABLE_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Saved detailed report: {REPORT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    _print_summary(inspect_datasets())