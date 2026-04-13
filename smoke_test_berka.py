import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


def build_mini_berka_trans(path: Path, n_users: int = 40, n_days: int = 140) -> None:
    rng = np.random.default_rng(42)
    rows = []
    start = pd.Timestamp("1993-01-01")
    operations = ["VKLAD", "PREVOD NA UCET", "VYBER", "VYBER KARTOU"]
    symbols = ["POJISTNE", "SLUZBY", "SIPO", "UROK", "DUCHOD", "UVER", ""]

    trans_id = 1
    for user in range(1, n_users + 1):
        account_id = 1000 + user
        for d in range(n_days):
            if rng.random() < 0.55:
                tx_per_day = rng.integers(1, 4)
                for _ in range(tx_per_day):
                    is_inflow = rng.random() < 0.35
                    tx_type = "PRIJEM" if is_inflow else "VYDAJ"
                    amount = float(np.round(max(5.0, rng.normal(600 if is_inflow else 450, 120)), 2))
                    date_val = int((start + pd.Timedelta(days=d)).strftime("%y%m%d"))
                    rows.append(
                        {
                            "trans_id": trans_id,
                            "account_id": account_id,
                            "date": date_val,
                            "type": tx_type,
                            "operation": str(rng.choice(operations)),
                            "amount": amount,
                            "balance": 0.0,
                            "k_symbol": str(rng.choice(symbols)),
                            "bank": "AB",
                            "account": account_id,
                        }
                    )
                    trans_id += 1

    pd.DataFrame(rows).to_csv(path, index=False, sep=";")


def run(cmd: list[str]) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    if result.stdout.strip():
        print(result.stdout.strip())


def main() -> None:
    root = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        raw_dir = tmp_path / "data" / "berka" / "raw"
        processed_dir = tmp_path / "data" / "berka" / "processed"
        weekly_dir = tmp_path / "step1"
        cls_out = tmp_path / "cls"
        reg_out = tmp_path / "reg"
        report_out = tmp_path / "report"
        raw_dir.mkdir(parents=True, exist_ok=True)

        build_mini_berka_trans(raw_dir / "trans.csv")

        run(["python3", str(root / "step0_berka_ingestion" / "run_berka_ingestion.py"), "--input-dir", str(raw_dir), "--output-dir", str(processed_dir), "--max-rows", "10000"])
        run(["python3", str(root / "step1_berka_weekly_builder" / "run_build_weekly.py"), "--input-csv", str(processed_dir / "transactions_normalized.csv"), "--output-dir", str(weekly_dir), "--test-weeks", "8"])
        run(["python3", str(root / "step3_model_training_berka" / "train_classification.py"), "--input-dir", str(weekly_dir / "classification"), "--output-dir", str(cls_out), "--quick"])
        run(["python3", str(root / "step3_regression_training" / "train_regression.py"), "--input-dir", str(weekly_dir / "regression"), "--output-dir", str(reg_out), "--quick"])
        run([
            "python3",
            str(root / "reports" / "berka_feasibility" / "generate_feasibility_report.py"),
            "--weekly-dir",
            str(weekly_dir),
            "--classification-report",
            str(cls_out / "classification_report_quick.json"),
            "--regression-report",
            str(reg_out / "regression_report_quick.json"),
            "--ingestion-report",
            str(processed_dir / "ingestion_report.json"),
            "--output-dir",
            str(report_out),
        ])

        expected = [
            processed_dir / "transactions_normalized.csv",
            weekly_dir / "classification" / "train_lag_features.csv",
            weekly_dir / "regression" / "train_regression.csv",
            cls_out / "classification_report_quick.json",
            reg_out / "regression_report_quick.json",
            report_out / "berka_feasibility_report.md",
        ]
        missing = [str(x) for x in expected if not x.exists()]
        if missing:
            raise RuntimeError(f"Smoke test failed. Missing: {missing}")

        print("Berka smoke test passed.")


if __name__ == "__main__":
    main()

