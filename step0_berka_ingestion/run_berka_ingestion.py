import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


INFLOW_TYPES = {"PRIJEM"}
OUTFLOW_TYPES = {"VYDAJ", "VYBER"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 0: Berka ingestion and normalization")
    parser.add_argument("--input-dir", type=str, default="data/berka/raw", help="Directory with raw Berka CSV files")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/berka/processed",
        help="Directory for normalized outputs",
    )
    parser.add_argument("--max-rows", type=int, default=0, help="Optional row cap for fast smoke runs")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_berka_date(series: pd.Series) -> pd.Series:
    raw = series.astype(str).str.extract(r"(\d{6,8})", expand=False)
    raw = raw.str.strip()

    parsed = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    mask6 = raw.str.len().eq(6)
    mask8 = raw.str.len().eq(8)

    # Berka uses YYMMDD; parse 6-digit values explicitly to keep years in 19xx/20xx range.
    parsed.loc[mask6] = pd.to_datetime(raw.loc[mask6], format="%y%m%d", errors="coerce")
    parsed.loc[mask8] = pd.to_datetime(raw.loc[mask8], format="%Y%m%d", errors="coerce")
    return parsed.dt.floor("D")


def flow_direction_from_type(type_series: pd.Series) -> pd.Series:
    normalized = type_series.fillna("").astype(str).str.upper().str.strip()
    direction = np.where(normalized.isin(INFLOW_TYPES), "inflow", "outflow")
    direction = np.where(normalized.isin(OUTFLOW_TYPES), "outflow", direction)
    return pd.Series(direction, index=type_series.index)


def build_category(df: pd.DataFrame) -> pd.Series:
    base = (
        df["k_symbol"].fillna("").astype(str).str.strip().replace("", pd.NA).fillna(
            df["operation"].fillna("").astype(str).str.strip().replace("", pd.NA)
        )
    )
    base = base.fillna(df["type"].fillna("").astype(str).str.strip()).replace("", "unknown")
    return df["flow_direction"].astype(str) + ":" + base.astype(str).str.lower()


def to_report_md(report: dict) -> str:
    top_categories = report.get("top_categories", [])
    top_lines = "\n".join([f"- `{row['category']}`: {row['count']}" for row in top_categories[:15]])
    return f"""# Berka Ingestion Report

- Rows read: **{report['rows_read']}**
- Rows normalized: **{report['rows_normalized']}**
- Unique users: **{report['unique_users']}**
- Date range: **{report['date_range_start']} -> {report['date_range_end']}**
- Missing rates: `{json.dumps(report['missing_rates'], ensure_ascii=False)}`

## Flow split
- Inflow rows: **{report['flow_counts'].get('inflow', 0)}**
- Outflow rows: **{report['flow_counts'].get('outflow', 0)}**

## Top categories
{top_lines if top_lines else '- n/a'}

## Category rule
`category = f"{{flow_direction}}:{{k_symbol or operation or type}}"`
"""


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    ensure_dir(output_dir)

    trans_path = input_dir / "trans.csv"
    if not trans_path.exists():
        raise FileNotFoundError(f"Missing required file: {trans_path}")

    df = pd.read_csv(trans_path, sep=";", low_memory=False)
    if args.max_rows > 0:
        df = df.head(args.max_rows).copy()

    required = ["account_id", "date", "type", "operation", "amount", "k_symbol"]
    missing_cols = [col for col in required if col not in df.columns]
    if missing_cols:
        raise RuntimeError(f"trans.csv missing columns: {missing_cols}")

    work = df.copy()
    work["transaction_date"] = parse_berka_date(work["date"])
    work["user_id"] = pd.to_numeric(work["account_id"], errors="coerce")
    work["amount_abs"] = pd.to_numeric(work["amount"], errors="coerce")
    work["flow_direction"] = flow_direction_from_type(work["type"])

    signed = np.where(work["flow_direction"].eq("inflow"), work["amount_abs"], -work["amount_abs"])
    work["amount"] = pd.to_numeric(signed, errors="coerce")
    work["category"] = build_category(work)

    normalized = work[
        [
            "user_id",
            "transaction_date",
            "amount",
            "amount_abs",
            "category",
            "flow_direction",
            "operation",
            "type",
            "k_symbol",
        ]
    ].copy()

    normalized = normalized.dropna(subset=["user_id", "transaction_date", "amount"])
    normalized["user_id"] = normalized["user_id"].astype(int)
    normalized["transaction_date"] = pd.to_datetime(normalized["transaction_date"], errors="coerce")
    normalized = normalized.sort_values(["transaction_date", "user_id"]).reset_index(drop=True)

    out_csv = output_dir / "transactions_normalized.csv"
    normalized.to_csv(out_csv, index=False)

    missing_rates = {
        col: float(normalized[col].isna().mean())
        for col in ["user_id", "transaction_date", "amount", "category", "flow_direction"]
    }
    top_categories = (
        normalized["category"].value_counts().head(30).rename_axis("category").reset_index(name="count").to_dict("records")
    )

    report = {
        "execution_timestamp": datetime.now(timezone.utc).isoformat(),
        "input_file": str(trans_path),
        "rows_read": int(len(df)),
        "rows_normalized": int(len(normalized)),
        "unique_users": int(normalized["user_id"].nunique()),
        "date_range_start": str(normalized["transaction_date"].min().date()),
        "date_range_end": str(normalized["transaction_date"].max().date()),
        "missing_rates": missing_rates,
        "flow_counts": {k: int(v) for k, v in normalized["flow_direction"].value_counts().to_dict().items()},
        "top_categories": top_categories,
        "category_rule": "category = f'{flow_direction}:{k_symbol or operation or type}'",
        "output_file": str(out_csv),
    }

    (output_dir / "ingestion_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "ingestion_report.md").write_text(to_report_md(report), encoding="utf-8")

    print("=== BERKA STEP 0 DONE ===")
    print(f"Rows: {len(normalized)} | Users: {normalized['user_id'].nunique()}")
    print(f"Output: {out_csv}")


if __name__ == "__main__":
    main()

