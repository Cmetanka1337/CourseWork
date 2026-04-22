import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_recall_fscore_support

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from on_device_calibrator.calibrator import SoftmaxCalibrator
from step3_model_training_berka.train_classification import CLASSIFICATION_FEATURES

LABELS = [0, 1, 2, 3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run per-user walk-forward personalization simulation with on-device calibrator")
    parser.add_argument(
        "--train-csv",
        type=str,
        default="step1_berka_weekly_builder/outputs/classification/train_lag_features.csv",
    )
    parser.add_argument(
        "--test-csv",
        type=str,
        default="step1_berka_weekly_builder/outputs/classification/test_lag_features.csv",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="step3_model_training_berka/outputs/full_spend_tuned_rf_model.pkl",
    )
    parser.add_argument("--target", type=str, default="bucket_spend_t_plus_1")
    parser.add_argument("--warmup-weeks", type=int, default=8)
    parser.add_argument("--update-every-weeks", type=int, default=2)
    parser.add_argument("--history-cap", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=1e-3)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--sgd-epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha-after-warmup", type=float, default=0.2)
    parser.add_argument("--report-json", type=str, default="reports/on_device_calibrator/calibrator_simulation_report.json")
    parser.add_argument("--report-md", type=str, default="reports/on_device_calibrator/calibrator_simulation_report.md")
    parser.add_argument("--plot-path", type=str, default="reports/on_device_calibrator/delta_f1_hist.png")
    return parser.parse_args()


def safe_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    if len(y_true) == 0:
        return {
            "f1_macro": 0.0,
            "balanced_accuracy": 0.0,
            "per_class_recall": {str(lbl): 0.0 for lbl in LABELS},
        }
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, zero_division=0)
    return {
        "f1_macro": float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "per_class_recall": {str(lbl): float(r[idx]) for idx, lbl in enumerate(LABELS)},
        "per_class_support": {str(lbl): int(s[idx]) for idx, lbl in enumerate(LABELS)},
    }


def user_level_deltas(eval_df: pd.DataFrame) -> dict:
    rows = []
    for user_id, part in eval_df.groupby("user_id", sort=False):
        uid = int(pd.to_numeric(part["user_id"].iloc[0]))
        y_true = part["y_true"].to_numpy(dtype=int)
        y_base = part["y_pred_base"].to_numpy(dtype=int)
        y_cal = part["y_pred_cal"].to_numpy(dtype=int)
        f1_base = float(f1_score(y_true, y_base, labels=LABELS, average="macro", zero_division=0))
        f1_cal = float(f1_score(y_true, y_cal, labels=LABELS, average="macro", zero_division=0))
        rows.append({"user_id": uid, "f1_base": f1_base, "f1_cal": f1_cal, "delta_f1": f1_cal - f1_base, "n": int(len(part))})

    df = pd.DataFrame(rows)
    if df.empty:
        return {
            "improved_users": 0,
            "worsened_users": 0,
            "unchanged_users": 0,
            "quantiles_delta_f1": {},
            "top_improved": [],
            "top_worsened": [],
        }

    eps = 1e-12
    improved = int((df["delta_f1"] > eps).sum())
    worsened = int((df["delta_f1"] < -eps).sum())
    unchanged = int(len(df) - improved - worsened)

    quantiles = {
        "p10": float(df["delta_f1"].quantile(0.10)),
        "p25": float(df["delta_f1"].quantile(0.25)),
        "p50": float(df["delta_f1"].quantile(0.50)),
        "p75": float(df["delta_f1"].quantile(0.75)),
        "p90": float(df["delta_f1"].quantile(0.90)),
    }

    top_improved = (
        df.sort_values("delta_f1", ascending=False)
        .head(10)[["user_id", "delta_f1", "f1_base", "f1_cal", "n"]]
        .to_dict(orient="records")
    )
    top_worsened = (
        df.sort_values("delta_f1", ascending=True)
        .head(10)[["user_id", "delta_f1", "f1_base", "f1_cal", "n"]]
        .to_dict(orient="records")
    )

    return {
        "improved_users": improved,
        "worsened_users": worsened,
        "unchanged_users": unchanged,
        "quantiles_delta_f1": quantiles,
        "top_improved": top_improved,
        "top_worsened": top_worsened,
        "per_user_rows": rows,
    }


def time_to_benefit(eval_df: pd.DataFrame) -> dict:
    results = []
    for user_id, part in eval_df.groupby("user_id", sort=False):
        uid = int(pd.to_numeric(part["user_id"].iloc[0]))
        part = part.sort_values("week_order").reset_index(drop=True)
        base_correct = np.asarray(part["y_pred_base"].to_numpy() == part["y_true"].to_numpy(), dtype=int)
        cal_correct = np.asarray(part["y_pred_cal"].to_numpy() == part["y_true"].to_numpy(), dtype=int)

        cum_base = np.cumsum(base_correct) / np.arange(1, len(base_correct) + 1)
        cum_cal = np.cumsum(cal_correct) / np.arange(1, len(cal_correct) + 1)
        better_idx = np.where(cum_cal > cum_base)[0]

        if better_idx.size == 0:
            results.append({"user_id": uid, "benefit_reached": False})
            continue

        idx = int(better_idx[0])
        results.append(
            {
                "user_id": uid,
                "benefit_reached": True,
                "weeks_until_benefit": int(idx + 1),
                "updates_until_benefit": int(part.loc[idx, "update_count"]),
            }
        )

    reached = [r for r in results if r.get("benefit_reached")]
    if not reached:
        return {"users_with_benefit": 0, "users_total": len(results), "median_updates_until_benefit": None, "median_weeks_until_benefit": None}

    return {
        "users_with_benefit": len(reached),
        "users_total": len(results),
        "median_updates_until_benefit": float(np.median([r["updates_until_benefit"] for r in reached])),
        "median_weeks_until_benefit": float(np.median([r["weeks_until_benefit"] for r in reached])),
    }


def build_markdown(payload: dict, plot_path: Path) -> str:
    overall = payload["overall"]
    per_user = payload["per_user_distribution"]
    stress = payload["small_data_stress_test"]
    ttb = payload["time_to_benefit"]

    return f"""# On-device Calibrator Simulation (Berka spend bucket)

Generated: {payload['execution_timestamp']}

## Setup
- Base model: `{payload['base_model_path']}`
- Target: `{payload['target']}`
- Walk-forward: per-user chronological simulation, no future labels in updates.
- Warm-up: first `{payload['config']['warmup_weeks']}` weeks (collect only).
- Update cadence: every `{payload['config']['update_every_weeks']}` weeks.
- Update data: cumulative per-user labeled buffer capped at `{payload['config']['history_cap']}` examples.

## Overall metrics on test timeline
| variant | f1_macro | balanced_accuracy |
| --- | ---: | ---: |
| RF base | {overall['base']['f1_macro']:.4f} | {overall['base']['balanced_accuracy']:.4f} |
| RF + calibrator | {overall['calibrated']['f1_macro']:.4f} | {overall['calibrated']['balanced_accuracy']:.4f} |
| RF + blended | {overall['blended']['f1_macro']:.4f} | {overall['blended']['balanced_accuracy']:.4f} |

## Per-user delta (calibrated - base, macro F1)
- Improved users: **{per_user['improved_users']}**
- Worsened users: **{per_user['worsened_users']}**
- Unchanged users: **{per_user['unchanged_users']}**
- Delta quantiles p10/p25/p50/p75/p90: **{per_user['quantiles_delta_f1']['p10']:.4f} / {per_user['quantiles_delta_f1']['p25']:.4f} / {per_user['quantiles_delta_f1']['p50']:.4f} / {per_user['quantiles_delta_f1']['p75']:.4f} / {per_user['quantiles_delta_f1']['p90']:.4f}**
- Histogram: `{plot_path}`

## Small-data stress test (users with 12-20 total weeks)
- Users: **{stress['users_count']}**
- Base F1 / balanced accuracy: **{stress['base']['f1_macro']:.4f} / {stress['base']['balanced_accuracy']:.4f}**
- Calibrated F1 / balanced accuracy: **{stress['calibrated']['f1_macro']:.4f} / {stress['calibrated']['balanced_accuracy']:.4f}**

## Time-to-benefit
- Users reaching benefit: **{ttb['users_with_benefit']} / {ttb['users_total']}**
- Median updates until benefit: **{ttb['median_updates_until_benefit']}**
- Median weeks until benefit: **{ttb['median_weeks_until_benefit']}**

## Swift port spec
- Formula: `p_adj = softmax(W * p_rf + b)` with `W` shape `4x4`, `b` shape `4`.
- Initialization: `W = I`, `b = 0` (identity behavior before updates).
- SGD update:
  - `logits = W * p_rf + b`
  - `p_adj = softmax(logits)`
  - `grad_logits = p_adj - one_hot(y)`
  - `grad_W = grad_logits outer p_rf + l2 * W`
  - `grad_b = grad_logits`
  - clip global grad norm to `clip`
  - `W -= lr * grad_W`, `b -= lr * grad_b`
- Recommended defaults:
  - `lr = {payload['config']['learning_rate']}`
  - `l2 = {payload['config']['l2']}`
  - `clip = {payload['config']['grad_clip']}`
  - update cadence: every `{payload['config']['update_every_weeks']}` weeks
  - warm-up: `{payload['config']['warmup_weeks']}` weeks
  - history cap: `{payload['config']['history_cap']}` examples
- On-device state to store:
  - calibrator params `W`, `b`
  - per-user ring buffer of `(p_rf, y_true, week_idx)` up to `K={payload['config']['history_cap']}`
  - counters: weeks since last update, updates count
"""


def main() -> None:
    args = parse_args()

    train_csv = Path(args.train_csv).resolve()
    test_csv = Path(args.test_csv).resolve()
    model_path = Path(args.model_path).resolve()
    report_json = Path(args.report_json).resolve()
    report_md = Path(args.report_md).resolve()
    plot_path = Path(args.plot_path).resolve()

    if not train_csv.exists() or not test_csv.exists():
        raise RuntimeError(
            "Weekly classification CSVs are required locally. Missing train/test files. "
            "Run step1_berka_weekly_builder/run_build_weekly.py first."
        )

    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    train_df["is_test"] = 0
    test_df["is_test"] = 1

    req_cols = ["user_id", "week_start", "week_t", args.target] + CLASSIFICATION_FEATURES
    missing_train = [c for c in req_cols if c not in train_df.columns]
    missing_test = [c for c in req_cols if c not in test_df.columns]
    if missing_train or missing_test:
        raise RuntimeError(f"Missing required columns. train:{missing_train} test:{missing_test}")

    combined = pd.concat([train_df, test_df], ignore_index=True)
    combined["week_start"] = pd.to_datetime(combined["week_start"], errors="coerce")
    combined = combined.sort_values(["user_id", "week_start", "week_t"]).reset_index(drop=True)
    combined["week_order"] = combined.groupby("user_id").cumcount() + 1

    model = joblib.load(model_path)
    x_all = combined[CLASSIFICATION_FEATURES].to_numpy(dtype=float)
    p_rf_all = model.predict_proba(x_all)
    classes = [int(c) for c in model.classes_]

    # Map class index for argmax back to class label.
    class_idx_to_label = {idx: cls for idx, cls in enumerate(classes)}

    eval_rows = []
    user_total_history = combined.groupby("user_id").size().to_dict()

    for user_id, part in combined.groupby("user_id", sort=False):
        uid = int(pd.to_numeric(part["user_id"].iloc[0]))
        part_idx = part.index.to_numpy()
        calibrator = SoftmaxCalibrator(
            n_classes=4,
            learning_rate=args.learning_rate,
            l2=args.l2,
            grad_clip_norm=args.grad_clip,
            seed=args.seed,
        )

        hist_probs = []
        hist_labels = []
        seen = 0
        since_last_update = 0
        update_count = 0

        for idx in part_idx:
            row = combined.loc[idx]
            p_rf = p_rf_all[idx].astype(float)
            p_cal = calibrator.predict_proba(p_rf)[0]

            alpha = args.alpha_after_warmup if seen >= args.warmup_weeks else 0.0
            p_blend = (1.0 - alpha) * p_rf + alpha * p_cal

            pred_base = int(class_idx_to_label[int(np.argmax(p_rf))])
            pred_cal = int(class_idx_to_label[int(np.argmax(p_cal))])
            pred_blend = int(class_idx_to_label[int(np.argmax(p_blend))])
            y_true = int(row[args.target])

            if int(row["is_test"]) == 1:
                eval_rows.append(
                    {
                        "user_id": uid,
                        "week_t": str(row["week_t"]),
                        "week_order": int(row["week_order"]),
                        "y_true": y_true,
                        "y_pred_base": pred_base,
                        "y_pred_cal": pred_cal,
                        "y_pred_blend": pred_blend,
                        "update_count": int(update_count),
                        "user_total_history": int(user_total_history.get(user_id, 0)),
                    }
                )

            # Label revealed after prediction -> append and potentially update.
            hist_probs.append(p_rf)
            hist_labels.append(y_true)
            if len(hist_probs) > args.history_cap:
                hist_probs = hist_probs[-args.history_cap :]
                hist_labels = hist_labels[-args.history_cap :]

            seen += 1
            since_last_update += 1

            if seen >= args.warmup_weeks and since_last_update >= args.update_every_weeks:
                calibrator.train_batch(np.asarray(hist_probs, dtype=float), np.asarray(hist_labels, dtype=int), epochs=args.sgd_epochs)
                update_count += 1
                since_last_update = 0

    eval_df = pd.DataFrame(eval_rows)

    y_true = eval_df["y_true"].to_numpy(dtype=int)
    y_base = eval_df["y_pred_base"].to_numpy(dtype=int)
    y_cal = eval_df["y_pred_cal"].to_numpy(dtype=int)
    y_blend = eval_df["y_pred_blend"].to_numpy(dtype=int)

    overall = {
        "base": safe_metrics(y_true, y_base),
        "calibrated": safe_metrics(y_true, y_cal),
        "blended": safe_metrics(y_true, y_blend),
    }

    per_user = user_level_deltas(eval_df)
    ttb = time_to_benefit(eval_df)

    stress_df = eval_df[(eval_df["user_total_history"] >= 12) & (eval_df["user_total_history"] <= 20)].copy()
    stress = {
        "users_count": int(stress_df["user_id"].nunique()),
        "rows": int(len(stress_df)),
        "base": safe_metrics(stress_df["y_true"].to_numpy(dtype=int), stress_df["y_pred_base"].to_numpy(dtype=int)),
        "calibrated": safe_metrics(stress_df["y_true"].to_numpy(dtype=int), stress_df["y_pred_cal"].to_numpy(dtype=int)),
        "blended": safe_metrics(stress_df["y_true"].to_numpy(dtype=int), stress_df["y_pred_blend"].to_numpy(dtype=int)),
    }

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    if per_user.get("per_user_rows"):
        per_user_df = pd.DataFrame(per_user["per_user_rows"])
        plt.figure(figsize=(7, 4))
        plt.hist(per_user_df["delta_f1"], bins=30, color="#4e79a7", edgecolor="black", alpha=0.85)
        plt.axvline(0.0, color="red", linestyle="--", linewidth=1)
        plt.title("Per-user delta macro-F1 (calibrated - base)")
        plt.xlabel("Delta macro-F1")
        plt.ylabel("Users")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=140)
        plt.close()

    payload = {
        "execution_timestamp": datetime.now(timezone.utc).isoformat(),
        "target": args.target,
        "base_model_path": str(model_path),
        "datasets": {"train_csv": str(train_csv), "test_csv": str(test_csv)},
        "config": {
            "warmup_weeks": int(args.warmup_weeks),
            "update_every_weeks": int(args.update_every_weeks),
            "history_cap": int(args.history_cap),
            "learning_rate": float(args.learning_rate),
            "l2": float(args.l2),
            "grad_clip": float(args.grad_clip),
            "sgd_epochs": int(args.sgd_epochs),
            "seed": int(args.seed),
            "alpha_after_warmup": float(args.alpha_after_warmup),
        },
        "evaluation_rows": int(len(eval_df)),
        "evaluation_users": int(eval_df["user_id"].nunique()),
        "overall": overall,
        "per_user_distribution": {k: v for k, v in per_user.items() if k != "per_user_rows"},
        "time_to_benefit": ttb,
        "small_data_stress_test": stress,
        "top_improved_users": per_user.get("top_improved", []),
        "top_worsened_users": per_user.get("top_worsened", []),
    }

    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    report_md.write_text(build_markdown(payload, plot_path), encoding="utf-8")

    print("Simulation report written:")
    print(report_json)
    print(report_md)


if __name__ == "__main__":
    main()

