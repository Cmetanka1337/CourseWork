import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


TARGET_CHOICES = ["bucket_spend_t_plus_1", "bucket_net_t_plus_1"]
LABELS = [0, 1, 2, 3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Berka classification training with time-based CV")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="step1_berka_weekly_builder/outputs/classification",
        help="Directory with train_lag_features.csv and test_lag_features.csv",
    )
    parser.add_argument("--output-dir", type=str, default="step3_model_training_berka/outputs")
    parser.add_argument("--target", type=str, default="bucket_spend_t_plus_1", choices=TARGET_CHOICES)
    parser.add_argument("--quick", action="store_true", help="Fast smoke mode")
    parser.add_argument("--save-prefix", type=str, default="", help="Optional prefix for report/model files")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    p, r, f1, s = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, zero_division=0)
    return {
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "per_class": {
            str(lbl): {
                "precision": float(p[idx]),
                "recall": float(r[idx]),
                "f1": float(f1[idx]),
                "support": int(s[idx]),
            }
            for idx, lbl in enumerate(LABELS)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
    }


def prepare_xy(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series]:
    features = [
        "bucket_spend_t",
        "bucket_net_t",
        "weekly_inflow_t",
        "weekly_outflow_t",
        "weekly_net_t",
        "txn_count_t",
        "category_diversity_t",
        "weekly_inflow_t_minus_1",
        "weekly_outflow_t_minus_1",
        "weekly_net_t_minus_1",
        "weekly_inflow_t_minus_2",
        "weekly_outflow_t_minus_2",
        "outflow_inflow_ratio_t",
    ]
    missing = [c for c in features + [target] if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")
    return df[features].copy(), df[target].astype(int)


def build_model_factories(is_quick: bool):
    rf_factory = lambda: RandomForestClassifier(
        random_state=42,
        n_estimators=120 if is_quick else 250,
        max_depth=12,
        min_samples_leaf=2,
        n_jobs=-1,
        class_weight="balanced",
    )
    lr_factory = lambda: Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    random_state=42,
                    max_iter=1200,
                    C=1.0,
                    class_weight="balanced",
                    solver="lbfgs",
                ),
            ),
        ]
    )
    sgd_factory = lambda: Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                SGDClassifier(
                    random_state=42,
                    loss="log_loss",
                    alpha=0.001,
                    max_iter=1500,
                    class_weight="balanced",
                ),
            ),
        ]
    )
    return {"random_forest": rf_factory, "logistic_regression": lr_factory, "sgd_classifier": sgd_factory}


def run_fold_metrics(model_factories: dict, x_train: pd.DataFrame, y_train: pd.Series, week_train: pd.Series, n_splits: int):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_rows = []
    class_rows = []
    confusion_payload = {name: [] for name in model_factories.keys()}

    for fold_idx, (tr_idx, va_idx) in enumerate(tscv.split(x_train), start=1):
        x_tr, x_va = x_train.iloc[tr_idx], x_train.iloc[va_idx]
        y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]
        period_start = str(week_train.iloc[va_idx].min())
        period_end = str(week_train.iloc[va_idx].max())

        for model_name, factory in model_factories.items():
            model = factory()
            model.fit(x_tr, y_tr)
            pred = model.predict(x_va)
            m = metrics(y_va.to_numpy(), pred)
            fold_rows.append(
                {
                    "fold": fold_idx,
                    "model": model_name,
                    "period_start": period_start,
                    "period_end": period_end,
                    "f1_macro": m["f1_macro"],
                    "balanced_accuracy": m["balanced_accuracy"],
                }
            )
            confusion_payload[model_name].append(
                {"fold": fold_idx, "period_start": period_start, "period_end": period_end, "matrix": m["confusion_matrix"]}
            )
            for cls, cls_m in m["per_class"].items():
                class_rows.append(
                    {
                        "fold": fold_idx,
                        "model": model_name,
                        "class": int(cls),
                        "precision": cls_m["precision"],
                        "recall": cls_m["recall"],
                        "f1": cls_m["f1"],
                        "support": cls_m["support"],
                    }
                )

    return pd.DataFrame(fold_rows), pd.DataFrame(class_rows), confusion_payload


def stability_summary(fold_df: pd.DataFrame, model_name: str) -> dict:
    model_df = fold_df[fold_df["model"] == model_name].sort_values("fold")
    values = model_df["f1_macro"].to_numpy(dtype=float)
    if len(values) < 3:
        return {"degradation": "insufficient_folds", "relative_drop": 0.0}
    head = float(np.mean(values[:2]))
    tail = float(np.mean(values[-2:]))
    relative_drop = float((head - tail) / max(abs(head), 1e-9))
    return {
        "degradation": "yes" if relative_drop > 0.10 else "no",
        "relative_drop": relative_drop,
        "head_mean_f1": head,
        "tail_mean_f1": tail,
    }


def save_stability_plot(fold_df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(8, 4))
    for model_name in ["random_forest", "logistic_regression", "sgd_classifier"]:
        model_df = fold_df[fold_df["model"] == model_name].sort_values("fold")
        if not model_df.empty:
            plt.plot(model_df["fold"], model_df["f1_macro"], marker="o", label=model_name)
    plt.xlabel("Fold")
    plt.ylabel("F1 macro")
    plt.title("Fold stability (chronological)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def prefixed(prefix: str, stem: str) -> str:
    return f"{prefix}_{stem}" if prefix else stem


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    ensure_dir(output_dir)

    run_mode = "quick" if args.quick else "full"
    target_tag = args.target.replace("bucket_", "").replace("_t_plus_1", "")
    auto_prefix = f"{run_mode}_{target_tag}"
    prefix = args.save_prefix.strip() if args.save_prefix.strip() else auto_prefix

    train_df = pd.read_csv(input_dir / "train_lag_features.csv").sort_values(["week_start", "user_id"]).reset_index(drop=True)
    test_df = pd.read_csv(input_dir / "test_lag_features.csv").sort_values(["week_start", "user_id"]).reset_index(drop=True)

    if args.quick:
        train_df = train_df.head(6000).copy()
        test_df = test_df.head(2000).copy()

    x_train, y_train = prepare_xy(train_df, args.target)
    x_test, y_test = prepare_xy(test_df, args.target)

    baseline_persist = metrics(y_test.to_numpy(), test_df[args.target.replace("_plus_1", "")].astype(int).to_numpy())
    majority_class = int(y_train.mode().iloc[0])
    baseline_majority = metrics(y_test.to_numpy(), np.full(len(y_test), majority_class, dtype=int))

    folds = 3 if args.quick else 5
    model_factories = build_model_factories(args.quick)

    fold_df, class_df, fold_confusions = run_fold_metrics(
        model_factories=model_factories,
        x_train=x_train,
        y_train=y_train,
        week_train=train_df["week_t"],
        n_splits=folds,
    )

    rf_model = model_factories["random_forest"]()
    rf_model.fit(x_train, y_train)
    lr_model = model_factories["logistic_regression"]()
    lr_model.fit(x_train, y_train)
    sgd_model = model_factories["sgd_classifier"]()
    sgd_model.fit(x_train, y_train)

    rf_test = metrics(y_test.to_numpy(), rf_model.predict(x_test))
    lr_test = metrics(y_test.to_numpy(), lr_model.predict(x_test))
    sgd_test = metrics(y_test.to_numpy(), sgd_model.predict(x_test))

    best_f1 = max(rf_test["f1_macro"], lr_test["f1_macro"], sgd_test["f1_macro"])
    baseline_f1 = baseline_persist["f1_macro"]
    gain_vs_persistence_relative = float((best_f1 - baseline_f1) / max(baseline_f1, 1e-9))

    report = {
        "execution_timestamp": datetime.now(timezone.utc).isoformat(),
        "run_mode": run_mode,
        "target": args.target,
        "rows": {"train": int(len(train_df)), "test": int(len(test_df))},
        "cv": {
            "strategy": f"TimeSeriesSplit(n_splits={folds})",
            "fold_metrics_file": prefixed(prefix, "fold_metrics.csv"),
            "fold_per_class_file": prefixed(prefix, "fold_per_class_metrics.csv"),
            "stability": {
                "random_forest": stability_summary(fold_df, "random_forest"),
                "logistic_regression": stability_summary(fold_df, "logistic_regression"),
                "sgd_classifier": stability_summary(fold_df, "sgd_classifier"),
            },
        },
        "acceptance": {
            "gain_formula": "(F1_model - F1_persistence) / F1_persistence",
            "relative_gain_vs_persistence": gain_vs_persistence_relative,
            "passes_50pct_gain": bool(gain_vs_persistence_relative >= 0.50),
            "passes_70pct_gain": bool(gain_vs_persistence_relative >= 0.70),
        },
        "test": {
            "baseline_persistence": baseline_persist,
            "baseline_majority": baseline_majority,
            "random_forest": rf_test,
            "logistic_regression": lr_test,
            "sgd_classifier": sgd_test,
        },
    }

    (output_dir / prefixed(prefix, "classification_report.json")).write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    fold_df.to_csv(output_dir / prefixed(prefix, "fold_metrics.csv"), index=False)
    class_df.to_csv(output_dir / prefixed(prefix, "fold_per_class_metrics.csv"), index=False)
    (output_dir / prefixed(prefix, "fold_confusion_matrices.json")).write_text(
        json.dumps(fold_confusions, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    save_stability_plot(fold_df, output_dir / prefixed(prefix, "stability_f1_by_fold.png"))

    if args.target == "bucket_spend_t_plus_1":
        (output_dir / f"classification_report_{run_mode}.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        fold_df.to_csv(output_dir / "fold_metrics.csv", index=False)

    joblib.dump(rf_model, output_dir / prefixed(prefix, "rf_model.pkl"))
    joblib.dump(lr_model, output_dir / prefixed(prefix, "lr_model.pkl"))
    joblib.dump(sgd_model, output_dir / prefixed(prefix, "sgd_model.pkl"))

    print("=== BERKA CLASSIFICATION DONE ===")
    print(f"Run mode: {run_mode} | Target: {args.target}")
    print(f"RF test F1 macro: {rf_test['f1_macro']:.4f}")


if __name__ == "__main__":
    main()

