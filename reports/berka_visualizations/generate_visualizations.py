import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from on_device_calibrator.calibrator import SoftmaxCalibrator
from step3_model_training_berka.train_classification import CLASSIFICATION_FEATURES

TARGET = "bucket_spend_t_plus_1"
PERSISTENCE_COL = "bucket_spend_t"
DEFAULT_CONFIG_REPORT = Path("reports/on_device_calibrator/calibrator_simulation_report.json")
DEFAULT_GOLDEN_SET = Path("models/release_candidate/golden_inference_set_full_spend_tuned.json")


@dataclass
class PersonalizationConfig:
    update_every_weeks: int = 2
    history_cap: int = 20
    learning_rate: float = 0.05
    l2: float = 1e-3
    grad_clip: float = 5.0
    sgd_epochs: int = 20
    seed: int = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Berka visualization pack for RF and blended calibrator")
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
        default="models/release_candidate/full_spend_tuned_rf_model.pkl",
    )
    parser.add_argument("--output-dir", type=str, default="reports/berka_visualizations")
    parser.add_argument("--alpha-after-warmup", type=float, default=0.2)
    parser.add_argument("--warmup-weeks", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def ensure_inputs(train_csv: Path, test_csv: Path, model_path: Path) -> None:
    missing = [str(p) for p in [train_csv, test_csv, model_path] if not p.exists()]
    if missing:
        details = "\n".join(f"  - {m}" for m in missing)
        raise RuntimeError(
            "Required input files are missing:\n"
            f"{details}\n"
            "Generate weekly CSVs with: python3 step1_berka_weekly_builder/run_build_weekly.py"
        )


def load_personalization_config(seed_override: int) -> PersonalizationConfig:
    cfg = PersonalizationConfig(seed=seed_override)
    if not DEFAULT_CONFIG_REPORT.exists():
        return cfg

    payload = json.loads(DEFAULT_CONFIG_REPORT.read_text(encoding="utf-8"))
    report_cfg = payload.get("config", {})
    return PersonalizationConfig(
        update_every_weeks=int(report_cfg.get("update_every_weeks", cfg.update_every_weeks)),
        history_cap=int(report_cfg.get("history_cap", cfg.history_cap)),
        learning_rate=float(report_cfg.get("learning_rate", cfg.learning_rate)),
        l2=float(report_cfg.get("l2", cfg.l2)),
        grad_clip=float(report_cfg.get("grad_clip", cfg.grad_clip)),
        sgd_epochs=int(report_cfg.get("sgd_epochs", cfg.sgd_epochs)),
        seed=seed_override,
    )


def safe_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return 0.0
    return float(np.mean(y_true == y_pred))


def safe_macro_f1(y_true: np.ndarray, y_pred: np.ndarray, labels: list[int]) -> float:
    if len(y_true) == 0:
        return 0.0
    return float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))


def plot_confusion(cm: np.ndarray, labels: list[int], title: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(np.arange(len(labels)), labels=labels)
    ax.set_yticks(np.arange(len(labels)), labels=labels)

    threshold = cm.max() * 0.6 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = int(cm[i, j])
            ax.text(j, i, str(value), ha="center", va="center", color="white" if value > threshold else "black")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_transition_accuracy(metrics_by_slice: dict[str, dict[str, float]], output_path: Path) -> None:
    groups = ["Transition weeks", "Stable weeks"]
    keys = ["transition", "stable"]
    model_names = ["RF", "Blended", "Persistence"]
    model_keys = ["rf", "blended", "persistence"]

    x = np.arange(len(groups))
    width = 0.24

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for idx, (m_name, m_key) in enumerate(zip(model_names, model_keys)):
        vals = [metrics_by_slice[k][m_key] for k in keys]
        ax.bar(x + (idx - 1) * width, vals, width=width, label=m_name)

    ax.set_xticks(x, labels=groups)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy on transition vs stability weeks")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def confidence_bin_report(df: pd.DataFrame, prob_col: str, pred_col: str) -> pd.DataFrame:
    edges = np.array([0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.001])
    labels = [f"{edges[i]:.2f}-{edges[i + 1]:.2f}" for i in range(len(edges) - 1)]

    part = df.copy()
    part["confidence_bin"] = pd.cut(part[prob_col], bins=edges, labels=labels, include_lowest=True, right=False)

    rows = []
    for label in labels:
        chunk = part[part["confidence_bin"] == label]
        n = int(len(chunk))
        acc = safe_accuracy(chunk["y_true"].to_numpy(dtype=int), chunk[pred_col].to_numpy(dtype=int))
        center = float(label.split("-")[0]) + 0.05
        rows.append({"bin": label, "center": center, "n": n, "accuracy": acc})
    return pd.DataFrame(rows)


def plot_confidence_curve(rf_bins: pd.DataFrame, blended_bins: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(rf_bins["center"], rf_bins["accuracy"], marker="o", label="RF")
    ax.plot(blended_bins["center"], blended_bins["accuracy"], marker="o", label="Blended")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence bin center (pmax)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Confidence vs accuracy")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_per_user_distribution(per_user_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.boxplot(
        [per_user_df["f1_rf"].to_numpy(), per_user_df["f1_blended"].to_numpy()],
        tick_labels=["RF", "Blended"],
        showmeans=True,
    )
    ax.set_ylim(0, 1)
    ax.set_ylabel("Per-user macro-F1")
    ax.set_title("Per-user macro-F1 distribution")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_delta_vs_history(per_user_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.scatter(per_user_df["n_weeks"], per_user_df["delta_f1"], alpha=0.6, s=20)
    ax.axhline(0.0, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("User history length in test (weeks)")
    ax.set_ylabel("Delta macro-F1 (blended - RF)")
    ax.set_title("User history length vs delta F1")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_user_trace(user_df: pd.DataFrame, user_id: int, output_path: Path) -> None:
    part = user_df.sort_values("week_start").reset_index(drop=True)
    x = np.arange(len(part))

    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.plot(x, part["y_true"], marker="o", label="True bucket", linewidth=2)
    ax.plot(x, part["y_pred_rf"], marker="s", label="RF prediction", alpha=0.85)
    ax.plot(x, part["y_pred_blended"], marker="^", label="Blended prediction", alpha=0.85)

    ax.set_title(f"User {user_id}: true vs predicted buckets over test weeks")
    ax.set_xlabel("Week index in test timeline")
    ax.set_ylabel("Bucket")
    ax.set_yticks([0, 1, 2, 3])
    ax.grid(alpha=0.25)
    ax.legend(ncols=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def run_golden_sanity(model, classes: list[int]) -> dict:
    if not DEFAULT_GOLDEN_SET.exists():
        return {"available": False, "reason": "golden_set_missing"}

    payload = json.loads(DEFAULT_GOLDEN_SET.read_text(encoding="utf-8"))
    records = payload.get("records", [])
    if not records:
        return {"available": False, "reason": "golden_set_empty"}

    golden_df = pd.DataFrame(records)
    missing_features = [c for c in CLASSIFICATION_FEATURES if c not in golden_df.columns]
    if missing_features:
        return {"available": False, "reason": f"missing_features: {missing_features}"}

    x_golden = golden_df[CLASSIFICATION_FEATURES].astype(float)
    proba = model.predict_proba(x_golden)
    pred_idx = np.argmax(proba, axis=1)
    pred_class = [int(classes[idx]) for idx in pred_idx]

    expected_pred = golden_df["predicted_class"].astype(int).to_numpy()
    match_rate = safe_accuracy(expected_pred, np.asarray(pred_class, dtype=int))

    return {
        "available": True,
        "records": int(len(golden_df)),
        "predicted_class_match_rate": match_rate,
    }


def pick_trace_users(eval_df: pd.DataFrame, seed: int) -> list[int]:
    counts = (
        eval_df.groupby("user_id")
        .size()
        .sort_values(ascending=False)
    )
    top_users = [int(uid) for uid in counts.head(3).index.tolist()]

    remaining = [int(uid) for uid in counts.index.tolist() if int(uid) not in top_users]
    rng = np.random.default_rng(seed)
    random_count = min(3, len(remaining))
    random_users = [] if random_count == 0 else [int(x) for x in rng.choice(remaining, size=random_count, replace=False)]

    selected = top_users + random_users
    if len(selected) < 6:
        for uid in counts.index.tolist():
            uid_int = int(uid)
            if uid_int not in selected:
                selected.append(uid_int)
            if len(selected) == 6:
                break
    return selected


def build_visualization_report(
    output_dir: Path,
    overall: dict,
    transition_metrics: dict,
    per_user_df: pd.DataFrame,
    trace_users: list[int],
    golden_sanity: dict,
    alpha_after_warmup: float,
    warmup_weeks: int,
) -> str:
    delta_mean = float(per_user_df["delta_f1"].mean()) if not per_user_df.empty else 0.0
    improved = int((per_user_df["delta_f1"] > 1e-12).sum()) if not per_user_df.empty else 0
    worsened = int((per_user_df["delta_f1"] < -1e-12).sum()) if not per_user_df.empty else 0

    trace_lines = "\n".join(f"![User {uid} trace](user_{uid}_trace.png)" for uid in trace_users)

    sanity_text = "Golden sanity: not available"
    if golden_sanity.get("available"):
        sanity_text = (
            "Golden sanity: predicted class match rate = "
            f"{golden_sanity['predicted_class_match_rate']:.3f} on {golden_sanity['records']} rows"
        )

    return f"""# Berka RF Visualization Report

## Configuration
- RF base model + blended calibrator with alpha={alpha_after_warmup} after warmup={warmup_weeks} weeks.
- Blending is evaluated in per-user chronological walk-forward mode.
- {sanity_text}

## Overall quality snapshot
| Variant | Accuracy | Macro-F1 |
| --- | ---: | ---: |
| RF base | {overall['rf']['accuracy']:.4f} | {overall['rf']['f1_macro']:.4f} |
| Blended | {overall['blended']['accuracy']:.4f} | {overall['blended']['f1_macro']:.4f} |
| Persistence | {overall['persistence']['accuracy']:.4f} | {overall['persistence']['f1_macro']:.4f} |

## A) Confusion matrices
- RF base:
![Confusion RF](confusion_rf.png)
- Blended:
![Confusion blended](confusion_blended.png)
- Persistence baseline:
![Confusion persistence](confusion_persistence.png)

Quick takeaway: heatmaps show whether predictions collapse into one bucket (one dominant column). RF vs blended vs persistence highlights where errors move.

## B) Time-series traces per user
{trace_lines}

Quick takeaway: traces show week-by-week behavior and where blended smoothing helps or misses true transitions.

## C) Transition-focused evaluation
![Transition accuracy](transition_accuracy.png)

- Transition weeks accuracy: RF={transition_metrics['transition']['rf']:.4f}, Blended={transition_metrics['transition']['blended']:.4f}, Persistence={transition_metrics['transition']['persistence']:.4f}
- Stable weeks accuracy: RF={transition_metrics['stable']['rf']:.4f}, Blended={transition_metrics['stable']['blended']:.4f}, Persistence={transition_metrics['stable']['persistence']:.4f}

## D) Confidence vs accuracy
![Confidence curve](confidence_curve.png)

Quick takeaway: this curve shows how pmax correlates with correctness; higher confidence should align with higher accuracy.

## E) Per-user behavior
![Per-user F1 distribution](per_user_f1_distribution.png)
![Delta F1 vs history](delta_f1_vs_history.png)

- Mean delta F1 (blended - RF): {delta_mean:.4f}
- Users improved: {improved}
- Users worsened: {worsened}

Also check `predictions_sample.csv` for a tabular sample across 5-10 users.
"""


def build_readme() -> str:
    return """# Berka Visualizations Pack

This pack visualizes bucket prediction quality for:
- RF base (release candidate)
- RF + blended calibrator (alpha after warmup)
- Persistence baseline (`bucket_t -> bucket_t+1`)

## What to open first
1. `confusion_rf.png`, `confusion_blended.png`, `confusion_persistence.png` - error structure and one-bucket collapse check.
2. `user_<id>_trace.png` - week-by-week behavior for specific users.
3. `transition_accuracy.png` - quality on transition vs stable weeks.
4. `confidence_curve.png` - confidence (pmax) vs accuracy relationship.
5. `per_user_f1_distribution.png` + `delta_f1_vs_history.png` - gain/loss distribution across users.

## Run
```bash
python3 reports/berka_visualizations/generate_visualizations.py --model-path models/release_candidate/full_spend_tuned_rf_model.pkl
```

## Main outputs
- `visualization_report.md`
- `predictions_sample.csv`
- PNG charts in the same folder
"""


def main() -> None:
    args = parse_args()
    train_csv = Path(args.train_csv).resolve()
    test_csv = Path(args.test_csv).resolve()
    model_path = Path(args.model_path).resolve()
    output_dir = Path(args.output_dir).resolve()

    ensure_inputs(train_csv, test_csv, model_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    personalization_cfg = load_personalization_config(seed_override=int(args.seed))

    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    train_df["is_test"] = 0
    test_df["is_test"] = 1

    required_columns = ["user_id", "week_start", "week_t", TARGET, PERSISTENCE_COL] + CLASSIFICATION_FEATURES
    missing_train = [col for col in required_columns if col not in train_df.columns]
    missing_test = [col for col in required_columns if col not in test_df.columns]
    if missing_train or missing_test:
        raise RuntimeError(f"Missing required columns. train:{missing_train} test:{missing_test}")

    combined = pd.concat([train_df, test_df], ignore_index=True)
    combined["week_start"] = pd.to_datetime(combined["week_start"], errors="coerce")
    combined = combined.sort_values(["user_id", "week_start", "week_t"]).reset_index(drop=True)

    model = joblib.load(model_path)
    proba_all = model.predict_proba(combined[CLASSIFICATION_FEATURES].astype(float))
    classes = [int(c) for c in model.classes_]

    eval_rows = []
    for user_id, part in combined.groupby("user_id", sort=False):
        calibrator = SoftmaxCalibrator(
            n_classes=len(classes),
            learning_rate=personalization_cfg.learning_rate,
            l2=personalization_cfg.l2,
            grad_clip_norm=personalization_cfg.grad_clip,
            seed=personalization_cfg.seed,
        )

        hist_probs: list[np.ndarray] = []
        hist_labels: list[int] = []
        seen = 0
        since_last_update = 0

        for idx in part.index.to_numpy():
            row = combined.loc[idx]
            p_rf = proba_all[idx].astype(float)
            p_cal = calibrator.predict_proba(p_rf)[0]

            alpha = args.alpha_after_warmup if seen >= args.warmup_weeks else 0.0
            p_blended = (1.0 - alpha) * p_rf + alpha * p_cal

            pred_rf = int(classes[int(np.argmax(p_rf))])
            pred_blended = int(classes[int(np.argmax(p_blended))])
            y_true = int(row[TARGET])
            bucket_t = int(row[PERSISTENCE_COL])

            if int(row["is_test"]) == 1:
                eval_rows.append(
                    {
                        "user_id": int(pd.to_numeric(row["user_id"])),
                        "week_start": str(pd.to_datetime(row["week_start"]).date()),
                        "week_t": str(row["week_t"]),
                        "y_true": y_true,
                        "bucket_t": bucket_t,
                        "y_pred_rf": pred_rf,
                        "y_pred_blended": pred_blended,
                        "pmax_rf": float(np.max(p_rf)),
                        "pmax_blended": float(np.max(p_blended)),
                        "was_transition": bool(y_true != bucket_t),
                    }
                )

            hist_probs.append(p_rf)
            hist_labels.append(y_true)
            if len(hist_probs) > personalization_cfg.history_cap:
                hist_probs = hist_probs[-personalization_cfg.history_cap :]
                hist_labels = hist_labels[-personalization_cfg.history_cap :]

            seen += 1
            since_last_update += 1

            if seen >= args.warmup_weeks and since_last_update >= personalization_cfg.update_every_weeks:
                calibrator.train_batch(
                    np.asarray(hist_probs, dtype=float),
                    np.asarray(hist_labels, dtype=int),
                    epochs=personalization_cfg.sgd_epochs,
                )
                since_last_update = 0

    eval_df = pd.DataFrame(eval_rows)
    if eval_df.empty:
        raise RuntimeError("Evaluation dataframe is empty. Check input datasets and target columns.")

    y_true = eval_df["y_true"].to_numpy(dtype=int)
    y_rf = eval_df["y_pred_rf"].to_numpy(dtype=int)
    y_blended = eval_df["y_pred_blended"].to_numpy(dtype=int)
    y_persistence = eval_df["bucket_t"].to_numpy(dtype=int)

    overall = {
        "rf": {"accuracy": safe_accuracy(y_true, y_rf), "f1_macro": safe_macro_f1(y_true, y_rf, classes)},
        "blended": {
            "accuracy": safe_accuracy(y_true, y_blended),
            "f1_macro": safe_macro_f1(y_true, y_blended, classes),
        },
        "persistence": {
            "accuracy": safe_accuracy(y_true, y_persistence),
            "f1_macro": safe_macro_f1(y_true, y_persistence, classes),
        },
    }

    cm_rf = confusion_matrix(y_true, y_rf, labels=classes)
    cm_blended = confusion_matrix(y_true, y_blended, labels=classes)
    cm_persistence = confusion_matrix(y_true, y_persistence, labels=classes)

    plot_confusion(cm_rf, classes, "Confusion matrix: RF base", output_dir / "confusion_rf.png")
    plot_confusion(cm_blended, classes, "Confusion matrix: RF + blended calibrator", output_dir / "confusion_blended.png")
    plot_confusion(cm_persistence, classes, "Confusion matrix: persistence baseline", output_dir / "confusion_persistence.png")

    transition_mask = eval_df["was_transition"].to_numpy(dtype=bool)
    stable_mask = ~transition_mask
    transition_metrics = {
        "transition": {
            "rf": safe_accuracy(y_true[transition_mask], y_rf[transition_mask]),
            "blended": safe_accuracy(y_true[transition_mask], y_blended[transition_mask]),
            "persistence": safe_accuracy(y_true[transition_mask], y_persistence[transition_mask]),
        },
        "stable": {
            "rf": safe_accuracy(y_true[stable_mask], y_rf[stable_mask]),
            "blended": safe_accuracy(y_true[stable_mask], y_blended[stable_mask]),
            "persistence": safe_accuracy(y_true[stable_mask], y_persistence[stable_mask]),
        },
    }
    plot_transition_accuracy(transition_metrics, output_dir / "transition_accuracy.png")

    rf_bins = confidence_bin_report(eval_df, prob_col="pmax_rf", pred_col="y_pred_rf")
    blended_bins = confidence_bin_report(eval_df, prob_col="pmax_blended", pred_col="y_pred_blended")
    plot_confidence_curve(rf_bins, blended_bins, output_dir / "confidence_curve.png")

    per_user_rows = []
    for user_id, part in eval_df.groupby("user_id"):
        y_u = part["y_true"].to_numpy(dtype=int)
        f1_rf = safe_macro_f1(y_u, part["y_pred_rf"].to_numpy(dtype=int), classes)
        f1_blended = safe_macro_f1(y_u, part["y_pred_blended"].to_numpy(dtype=int), classes)
        per_user_rows.append(
            {
                "user_id": int(pd.to_numeric(user_id)),
                "n_weeks": int(len(part)),
                "f1_rf": f1_rf,
                "f1_blended": f1_blended,
                "delta_f1": f1_blended - f1_rf,
            }
        )

    per_user_df = pd.DataFrame(per_user_rows)
    plot_per_user_distribution(per_user_df, output_dir / "per_user_f1_distribution.png")
    plot_delta_vs_history(per_user_df, output_dir / "delta_f1_vs_history.png")

    trace_users = pick_trace_users(eval_df, seed=int(args.seed))
    for uid in trace_users:
        plot_user_trace(eval_df[eval_df["user_id"] == uid].copy(), uid, output_dir / f"user_{uid}_trace.png")

    sample_cols = [
        "user_id",
        "week_start",
        "y_true",
        "y_pred_rf",
        "y_pred_blended",
        "pmax_rf",
        "pmax_blended",
        "was_transition",
    ]
    eval_df[eval_df["user_id"].isin(trace_users)][sample_cols].to_csv(output_dir / "predictions_sample.csv", index=False)

    golden_sanity = run_golden_sanity(model=model, classes=classes)

    report_md = build_visualization_report(
        output_dir=output_dir,
        overall=overall,
        transition_metrics=transition_metrics,
        per_user_df=per_user_df,
        trace_users=trace_users,
        golden_sanity=golden_sanity,
        alpha_after_warmup=float(args.alpha_after_warmup),
        warmup_weeks=int(args.warmup_weeks),
    )
    (output_dir / "visualization_report.md").write_text(report_md, encoding="utf-8")
    (output_dir / "README.md").write_text(build_readme(), encoding="utf-8")

    summary_payload = {
        "overall": overall,
        "transition_metrics": transition_metrics,
        "trace_users": trace_users,
        "golden_sanity": golden_sanity,
        "config": {
            "alpha_after_warmup": float(args.alpha_after_warmup),
            "warmup_weeks": int(args.warmup_weeks),
            "seed": int(args.seed),
            "personalization": personalization_cfg.__dict__,
        },
    }
    (output_dir / "visualization_summary.json").write_text(
        json.dumps(summary_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("Visualization pack generated:")
    print(output_dir)


if __name__ == "__main__":
    main()


