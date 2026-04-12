import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


def build_synthetic_csv(path: Path) -> None:
    rng = np.random.default_rng(42)
    users = ["u1", "u2", "u3"]
    merchants = ["Fresh Market", "Gas Station", "Coffee Bar", "Online Store"]
    days = pd.date_range("2024-01-01", periods=45, freq="D", tz="UTC")

    rows = []
    for user in users:
        base = rng.uniform(5, 40)
        for day in days:
            for merchant in merchants:
                if rng.uniform() < 0.45:
                    amount = round(max(0.0, rng.normal(base, 8)), 2)
                    rows.append(
                        {
                            "user_id": user,
                            "timestamp": day.isoformat(),
                            "amount": amount,
                            "merchant_name": merchant,
                        }
                    )

    pd.DataFrame(rows).to_csv(path, index=False)


def main() -> None:
    project_root = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        csv_path = tmp_path / "synthetic.csv"
        out_path = tmp_path / "outputs"
        build_synthetic_csv(csv_path)

        cmd = [
            "python3",
            str(project_root / "main.py"),
            "--csv-path",
            str(csv_path),
            "--user-col",
            "user_id",
            "--date-col",
            "timestamp",
            "--amount-col",
            "amount",
            "--category-col",
            "merchant_name",
            "--collapse-mode",
            "manual",
            "--output-dir",
            str(out_path),
        ]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout.strip())

        expected = [
            out_path / "report.md",
            out_path / "report.json",
            out_path / "tables" / "stage3_sparsity.csv",
            out_path / "tables" / "stage5_autocorrelation.csv",
            out_path / "figures" / "stage4_hist_daily_spend.png",
        ]
        missing = [str(p) for p in expected if not p.exists()]
        if missing:
            raise RuntimeError(f"Smoke test failed, missing artifacts: {missing}")

        print("Smoke test passed: pipeline generated expected artifacts.")


if __name__ == "__main__":
    main()

