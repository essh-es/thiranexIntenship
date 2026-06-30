"""
Data Cleaning & Reporting Automation Tool
==========================================
Usage:
    python data_cleaner.py <input_file> [--output OUTPUT_DIR] [--sheet SHEET_NAME]

Supports .csv, .xlsx, .xls input.
Produces:
    - <name>_cleaned.xlsx   (cleaned data)
    - <name>_report.xlsx    (summary stats + charts)
    - <name>_log.txt        (every cleaning action taken, for audit trail)
"""

import argparse
import sys
import os
from datetime import datetime

import pandas as pd
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


class CleaningLog:
    """Tracks every transformation applied, so the process is auditable."""
    def __init__(self):
        self.entries = []

    def add(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.entries.append(f"[{timestamp}] {message}")
        print(message)

    def save(self, path):
        with open(path, "w") as f:
            f.write("\n".join(self.entries))


def load_data(filepath, sheet_name=None):
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(filepath)
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(filepath, sheet_name=sheet_name or 0)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    return df


def profile_data(df, log):
    """Return a profiling DataFrame: dtype, missing %, unique count, duplicates."""
    log.add(f"Profiling dataset: {df.shape[0]} rows x {df.shape[1]} columns")
    profile = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing_count": df.isna().sum(),
        "missing_pct": (df.isna().mean() * 100).round(2),
        "unique_values": df.nunique(),
    })
    n_dupes = df.duplicated().sum()
    log.add(f"Found {n_dupes} fully duplicate rows")
    return profile, n_dupes


def clean_text_columns(df, log):
    """Trim whitespace, normalize case-inconsistent categories, standardize blanks."""
    obj_cols = df.select_dtypes(include="object").columns
    for col in obj_cols:
        before_blanks = df[col].isin(["", " ", "NA", "N/A", "null", "NULL", "none", "None", "-"]).sum()
        df[col] = df[col].replace(
            ["", " ", "NA", "N/A", "null", "NULL", "none", "None", "-"], np.nan
        )
        if df[col].dtype == object:
            df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
        if before_blanks > 0:
            log.add(f"Column '{col}': normalized {before_blanks} blank/placeholder values to NaN")

        # Detect case-inconsistent categorical values (e.g. "North" vs "north")
        # Only applied to low-cardinality text columns, treated as categories.
        non_null = df[col].dropna()
        if len(non_null) > 0 and non_null.nunique() <= 30:
            distinct_raw = non_null.astype(str).nunique()
            distinct_lower = non_null.astype(str).str.lower().nunique()
            if distinct_lower < distinct_raw:
                canonical = (
                    non_null.astype(str)
                    .groupby(non_null.astype(str).str.lower())
                    .agg(lambda s: s.value_counts().idxmax())
                )
                mapping = {v: canonical[v.lower()] for v in non_null.astype(str).unique()}
                n_changed = sum(1 for v in non_null.astype(str) if mapping[v] != v)
                if n_changed > 0:
                    df[col] = df[col].apply(
                        lambda x: mapping.get(str(x), x) if pd.notna(x) else x
                    )
                    log.add(f"Column '{col}': standardized casing for {n_changed} values "
                             f"({distinct_raw} variants -> {distinct_lower} categories)")
    return df


def handle_missing_values(df, log, numeric_strategy="median", categorical_strategy="mode"):
    """
    Fill missing numeric values with median (robust to outliers) and
    missing categorical values with mode, unless missingness is too high
    (>50%), in which case the column is flagged rather than imputed.
    """
    for col in df.columns:
        missing_pct = df[col].isna().mean() * 100
        if missing_pct == 0:
            continue
        if missing_pct > 50:
            log.add(f"Column '{col}': {missing_pct:.1f}% missing — too sparse to impute reliably, left as-is (consider dropping)")
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            fill_value = df[col].median()
            df[col] = df[col].fillna(fill_value)
            log.add(f"Column '{col}': filled {missing_pct:.1f}% missing numeric values with median ({fill_value:.2f})")
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            log.add(f"Column '{col}': {missing_pct:.1f}% missing dates — left as NaT (no safe default)")
        else:
            mode_vals = df[col].mode()
            if not mode_vals.empty:
                fill_value = mode_vals[0]
                df[col] = df[col].fillna(fill_value)
                log.add(f"Column '{col}': filled {missing_pct:.1f}% missing values with mode ('{fill_value}')")
    return df


def remove_duplicates(df, log):
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    removed = before - len(df)
    if removed > 0:
        log.add(f"Removed {removed} duplicate rows ({before} -> {len(df)})")
    else:
        log.add("No duplicate rows removed")
    return df


def standardize_dates(df, log):
    """Attempt to detect and standardize date-like columns to ISO format."""
    for col in df.columns:
        if df[col].dtype == object:
            sample = df[col].dropna().astype(str).head(20)
            if len(sample) == 0:
                continue
            date_like = sample.str.match(
                r"^\d{1,4}[-/]\d{1,2}[-/]\d{1,4}"
            ).mean()
            if date_like > 0.7:
                try:
                    converted = pd.to_datetime(df[col], errors="coerce")
                    failed = converted.isna().sum() - df[col].isna().sum()
                    if failed / max(len(df), 1) < 0.2:
                        df[col] = converted
                        log.add(f"Column '{col}': standardized to datetime format ({failed} values could not be parsed)")
                except Exception:
                    pass
    return df


def detect_outliers(df, log):
    """Flag numeric outliers using IQR method; reports only, does not remove."""
    numeric_cols = df.select_dtypes(include=np.number).columns
    outlier_summary = {}
    for col in numeric_cols:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        if n_outliers > 0:
            outlier_summary[col] = n_outliers
            log.add(f"Column '{col}': {n_outliers} potential outliers detected (outside [{lower:.2f}, {upper:.2f}])")
    return outlier_summary


def generate_charts(df, profile, output_dir, base_name):
    """Generate PNG charts summarizing the dataset; returns list of file paths."""
    if plt is None:
        return []
    chart_paths = []

    # Missing values bar chart
    missing = profile["missing_pct"][profile["missing_pct"] > 0].sort_values(ascending=False)
    if len(missing) > 0:
        fig, ax = plt.subplots(figsize=(8, max(3, len(missing) * 0.4)))
        missing.plot(kind="barh", ax=ax, color="#d9534f")
        ax.set_xlabel("% Missing")
        ax.set_title("Missing Values by Column")
        plt.tight_layout()
        path = os.path.join(output_dir, f"{base_name}_missing_values.png")
        fig.savefig(path, dpi=120)
        plt.close(fig)
        chart_paths.append(path)

    # Distribution of numeric columns
    numeric_cols = df.select_dtypes(include=np.number).columns[:6]
    if len(numeric_cols) > 0:
        n = len(numeric_cols)
        cols = 2
        rows = (n + 1) // 2
        fig, axes = plt.subplots(rows, cols, figsize=(10, 3.5 * rows))
        axes = np.array(axes).reshape(-1)
        for i, col in enumerate(numeric_cols):
            df[col].dropna().plot(kind="hist", bins=20, ax=axes[i], color="#5bc0de", edgecolor="white")
            axes[i].set_title(col)
        for j in range(len(numeric_cols), len(axes)):
            fig.delaxes(axes[j])
        plt.tight_layout()
        path = os.path.join(output_dir, f"{base_name}_distributions.png")
        fig.savefig(path, dpi=120)
        plt.close(fig)
        chart_paths.append(path)

    return chart_paths


def export_report(df, profile, n_dupes_original, outlier_summary, chart_paths, output_path):
    """Write cleaned data + summary stats into a multi-sheet Excel report with embedded charts."""
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Cleaned Data", index=False)
        profile.to_excel(writer, sheet_name="Column Profile")

        summary_rows = [
            ("Total rows", len(df)),
            ("Total columns", len(df.columns)),
            ("Duplicate rows removed", int(n_dupes_original)),
            ("Columns with outliers", len(outlier_summary)),
            ("Report generated", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        ]
        pd.DataFrame(summary_rows, columns=["Metric", "Value"]).to_excel(
            writer, sheet_name="Summary", index=False
        )

        if outlier_summary:
            pd.DataFrame(
                list(outlier_summary.items()), columns=["Column", "Outlier Count"]
            ).to_excel(writer, sheet_name="Outliers", index=False)

    # Embed charts into the Summary sheet
    if chart_paths:
        from openpyxl import load_workbook
        from openpyxl.drawing.image import Image as XLImage

        wb = load_workbook(output_path)
        ws = wb["Summary"]
        row_cursor = 10
        for chart_path in chart_paths:
            img = XLImage(chart_path)
            img.anchor = f"A{row_cursor}"
            ws.add_image(img)
            row_cursor += 22
        wb.save(output_path)


def run_pipeline(input_file, output_dir="output", sheet_name=None):
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    log = CleaningLog()

    log.add(f"=== Data Cleaning & Reporting Pipeline started for '{input_file}' ===")
    df = load_data(input_file, sheet_name)
    log.add(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")

    profile_before, n_dupes_original = profile_data(df, log)

    df = clean_text_columns(df, log)
    df = standardize_dates(df, log)
    df = handle_missing_values(df, log)
    df = remove_duplicates(df, log)
    outlier_summary = detect_outliers(df, log)

    profile_after, _ = profile_data(df, log)

    cleaned_path = os.path.join(output_dir, f"{base_name}_cleaned.xlsx")
    df.to_excel(cleaned_path, index=False)
    log.add(f"Cleaned data saved to {cleaned_path}")

    chart_paths = generate_charts(df, profile_after, output_dir, base_name)

    report_path = os.path.join(output_dir, f"{base_name}_report.xlsx")
    export_report(df, profile_after, n_dupes_original, outlier_summary, chart_paths, report_path)
    log.add(f"Report saved to {report_path}")

    log_path = os.path.join(output_dir, f"{base_name}_log.txt")
    log.save(log_path)
    log.add(f"Cleaning log saved to {log_path}")
    log.add("=== Pipeline complete ===")

    return {
        "cleaned_path": cleaned_path,
        "report_path": report_path,
        "log_path": log_path,
    }


def main():
    parser = argparse.ArgumentParser(description="Automate data cleaning and reporting.")
    parser.add_argument("input_file", help="Path to input CSV or Excel file")
    parser.add_argument("--output", default="output", help="Output directory")
    parser.add_argument("--sheet", default=None, help="Sheet name (for Excel input with multiple sheets)")
    args = parser.parse_args()

    run_pipeline(args.input_file, args.output, args.sheet)


if __name__ == "__main__":
    main()
