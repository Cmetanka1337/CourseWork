import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import ElasticNet, Ridge, SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Berka regression training (single model over categories)")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="step1_berka_weekly_builder/outputs/regression",
        help="Directory with train_regression.csv and test_regression.csv",
    )
    parser.add_argument("--output-dir", type=str, default="step3_regression_training/outputs")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--save-prefix", type=str, default="", help="Optional prefix for report/model files")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def eval_reg(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mask = y_true > 0
    if mask.any():
        mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)
    else:
        mape = None
    return {"mae": mae, "rmse": rmse, "mape_positive_only": mape}


def build_preprocessor() -> ColumnTransformer:
    numeric = [
        "amount_cat_t",
        "txn_count_cat_t",
        "amount_cat_t_minus_1",
        "amount_cat_t_minus_2",
        "rolling_mean_4",
        "rolling_std_4",
        "weekly_inflow_t_minus_1",
        "weekly_outflow_t_minus_1",
        "weekly_net_t_minus_1",
    ]
    categorical = ["category", "flow_direction"]
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )


def cv_mae(model: Pipeline, x: pd.DataFrame, y: pd.Series, n_splits: int) -> list[float]:
    tscv = TimeSeriesSplit(n_splits=n_splits)
    out = []
    for tr_idx, va_idx in tscv.split(x):
        x_tr, x_va = x.iloc[tr_idx], x.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
        model.fit(x_tr, y_tr)
        pred = model.predict(x_va)
        out.append(float(mean_absolute_error(y_va, pred)))
    return out


def run_fold_metrics(models: dict[str, Pipeline], x: pd.DataFrame, y: pd.Series, week_t: pd.Series, n_splits: int) -> pd.DataFrame:
    tscv = TimeSeriesSplit(n_splits=n_splits)
    rows = []
    for fold_idx, (tr_idx, va_idx) in enumerate(tscv.split(x), start=1):
        x_tr, x_va = x.iloc[tr_idx], x.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
        period_start = str(week_t.iloc[va_idx].min())
        period_end = str(week_t.iloc[va_idx].max())
        for model_name, model in models.items():
            model.fit(x_tr, y_tr)
            pred = model.predict(x_va)
            rows.append(
                {
                    "fold": fold_idx,
                    "model": model_name,
                    "period_start": period_start,
                    "period_end": period_end,
                    "mae": float(mean_absolute_error(y_va, pred)),
                    "rmse": float(np.sqrt(mean_squared_error(y_va, pred))),
                }
            )
    return pd.DataFrame(rows)


def regression_stability(fold_df: pd.DataFrame, model_name: str) -> dict:
    model_df = fold_df[fold_df["model"] == model_name].sort_values("fold")
    values = model_df["mae"].to_numpy(dtype=float)
    if len(values) < 3:
        return {"degradation": "insufficient_folds", "relative_increase": 0.0}
    head = float(np.mean(values[:2]))
    tail = float(np.mean(values[-2:]))
    relative_increase = float((tail - head) / max(abs(head), 1e-9))
    return {
        "degradation": "yes" if relative_increase > 0.10 else "no",
        "relative_increase": relative_increase,
        "head_mean_mae": head,
        "tail_mean_mae": tail,
    }


def save_regression_stability_plot(fold_df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(8, 4))
    for model_name in ["ridge", "elasticnet", "sgd_regressor"]:
        model_df = fold_df[fold_df["model"] == model_name].sort_values("fold")
        if not model_df.empty:
            plt.plot(model_df["fold"], model_df["mae"], marker="o", label=model_name)
    plt.xlabel("Fold")
    plt.ylabel("MAE")
    plt.title("Regression fold stability (chronological)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def prefixed(prefix: str, stem: str) -> str:
    return f"{prefix}_{stem}" if prefix else stem


def top_category_breakdown(df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, top_n: int = 8) -> list[dict]:
    eval_df = pd.DataFrame({"category": df["category"].values, "y_true": y_true, "y_pred": y_pred})
    top_cats = eval_df["category"].value_counts().head(top_n).index
    rows = []
    for cat in top_cats:
        part = eval_df[eval_df["category"] == cat]
        rows.append(
            {
                "category": str(cat),
                "support": int(len(part)),
                "mae": float(mean_absolute_error(part["y_true"], part["y_pred"])),
                "rmse": float(np.sqrt(mean_squared_error(part["y_true"], part["y_pred"]))),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    ensure_dir(output_dir)
    run_mode = "quick" if args.quick else "full"
    prefix = args.save_prefix.strip() if args.save_prefix.strip() else run_mode

    train_df = pd.read_csv(input_dir / "train_regression.csv").sort_values(["week_start", "user_id", "category"]).reset_index(drop=True)
    test_df = pd.read_csv(input_dir / "test_regression.csv").sort_values(["week_start", "user_id", "category"]).reset_index(drop=True)

    if args.quick:
        train_df = train_df.head(12000).copy()
        test_df = test_df.head(4000).copy()

    target_col = "amount_cat_t_plus_1"
    feature_cols = [
        "amount_cat_t",
        "txn_count_cat_t",
        "amount_cat_t_minus_1",
        "amount_cat_t_minus_2",
        "rolling_mean_4",
        "rolling_std_4",
        "weekly_inflow_t_minus_1",
        "weekly_outflow_t_minus_1",
        "weekly_net_t_minus_1",
        "category",
        "flow_direction",
    ]

    missing = [c for c in feature_cols + [target_col] if c not in train_df.columns]
    if missing:
        raise RuntimeError(f"Missing columns in regression data: {missing}")

    x_train = train_df[feature_cols].copy()
    y_train = train_df[target_col].astype(float)
    x_test = test_df[feature_cols].copy()
    y_test = test_df[target_col].astype(float)

    baseline_last = x_test["amount_cat_t"].to_numpy(dtype=float)
    baseline_roll = x_test["rolling_mean_4"].to_numpy(dtype=float)

    base_last_metrics = eval_reg(y_test.to_numpy(), baseline_last)
    base_roll_metrics = eval_reg(y_test.to_numpy(), baseline_roll)

    pre = build_preprocessor()
    ridge = Pipeline(steps=[("prep", pre), ("model", Ridge(alpha=1.0, random_state=42))])
    enet = Pipeline(steps=[("prep", pre), ("model", ElasticNet(alpha=0.001, l1_ratio=0.2, random_state=42, max_iter=5000))])
    sgd = Pipeline(
        steps=[
            ("prep", pre),
            ("model", SGDRegressor(random_state=42, alpha=0.0005, penalty="l2", max_iter=3000, tol=1e-4)),
        ]
    )

    folds = 2 if args.quick else 3
    ridge_cv = cv_mae(ridge, x_train, y_train, folds)
    enet_cv = cv_mae(enet, x_train, y_train, folds)
    sgd_cv = cv_mae(sgd, x_train, y_train, folds)

    fold_metrics_df = run_fold_metrics(
        models={
            "ridge": Pipeline(steps=[("prep", build_preprocessor()), ("model", Ridge(alpha=1.0, random_state=42))]),
            "elasticnet": Pipeline(
                steps=[("prep", build_preprocessor()), ("model", ElasticNet(alpha=0.001, l1_ratio=0.2, random_state=42, max_iter=5000))]
            ),
            "sgd_regressor": Pipeline(
                steps=[("prep", build_preprocessor()), ("model", SGDRegressor(random_state=42, alpha=0.0005, penalty="l2", max_iter=3000, tol=1e-4))]
            ),
        },
        x=x_train,
        y=y_train,
        week_t=train_df["week_t"],
        n_splits=folds,
    )

    ridge.fit(x_train, y_train)
    enet.fit(x_train, y_train)
    sgd.fit(x_train, y_train)

    ridge_pred = ridge.predict(x_test)
    enet_pred = enet.predict(x_test)
    sgd_pred = sgd.predict(x_test)

    ridge_metrics = eval_reg(y_test.to_numpy(), ridge_pred)
    enet_metrics = eval_reg(y_test.to_numpy(), enet_pred)
    sgd_metrics = eval_reg(y_test.to_numpy(), sgd_pred)

    persistence_mae = base_last_metrics["mae"]
    ridge_metrics["skill_mae_vs_persistence"] = float(ridge_metrics["mae"] / max(persistence_mae, 1e-9))
    enet_metrics["skill_mae_vs_persistence"] = float(enet_metrics["mae"] / max(persistence_mae, 1e-9))
    sgd_metrics["skill_mae_vs_persistence"] = float(sgd_metrics["mae"] / max(persistence_mae, 1e-9))

    report = {
        "execution_timestamp": datetime.now(timezone.utc).isoformat(),
        "run_mode": run_mode,
        "cv_strategy": f"TimeSeriesSplit(n_splits={folds})",
        "rows": {"train": int(len(train_df)), "test": int(len(test_df))},
        "baselines_test": {
            "persistence_last_value": base_last_metrics,
            "rolling_mean_4": base_roll_metrics,
        },
        "models_test": {
            "ridge": ridge_metrics,
            "elasticnet": enet_metrics,
            "sgd_regressor": sgd_metrics,
        },
        "cv_mae": {
            "ridge": ridge_cv,
            "elasticnet": enet_cv,
            "sgd_regressor": sgd_cv,
        },
        "cv_fold_metrics_file": prefixed(prefix, "fold_metrics.csv"),
        "stability": {
            "ridge": regression_stability(fold_metrics_df, "ridge"),
            "elasticnet": regression_stability(fold_metrics_df, "elasticnet"),
            "sgd_regressor": regression_stability(fold_metrics_df, "sgd_regressor"),
        },
        "acceptance": {
            "gain_formula": "(MAE_persistence - MAE_model) / MAE_persistence",
            "best_improvement": float(
                max(
                    (base_last_metrics["mae"] - ridge_metrics["mae"]) / max(base_last_metrics["mae"], 1e-9),
                    (base_last_metrics["mae"] - enet_metrics["mae"]) / max(base_last_metrics["mae"], 1e-9),
                    (base_last_metrics["mae"] - sgd_metrics["mae"]) / max(base_last_metrics["mae"], 1e-9),
                )
            ),
            "passes_50pct_gain": bool(
                max(
                    (base_last_metrics["mae"] - ridge_metrics["mae"]) / max(base_last_metrics["mae"], 1e-9),
                    (base_last_metrics["mae"] - enet_metrics["mae"]) / max(base_last_metrics["mae"], 1e-9),
                    (base_last_metrics["mae"] - sgd_metrics["mae"]) / max(base_last_metrics["mae"], 1e-9),
                )
                >= 0.50
            ),
            "passes_70pct_gain": bool(
                max(
                    (base_last_metrics["mae"] - ridge_metrics["mae"]) / max(base_last_metrics["mae"], 1e-9),
                    (base_last_metrics["mae"] - enet_metrics["mae"]) / max(base_last_metrics["mae"], 1e-9),
                    (base_last_metrics["mae"] - sgd_metrics["mae"]) / max(base_last_metrics["mae"], 1e-9),
                )
                >= 0.70
            ),
        },
        "top_category_metrics_ridge": top_category_breakdown(test_df, y_test.to_numpy(), ridge_pred, top_n=10),
    }

    (output_dir / prefixed(prefix, "regression_report.json")).write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / f"regression_report_{run_mode}.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    fold_metrics_df.to_csv(output_dir / prefixed(prefix, "fold_metrics.csv"), index=False)
    save_regression_stability_plot(fold_metrics_df, output_dir / prefixed(prefix, "stability_mae_by_fold.png"))
    joblib.dump(ridge, output_dir / prefixed(prefix, "ridge_model.pkl"))
    joblib.dump(enet, output_dir / prefixed(prefix, "elasticnet_model.pkl"))
    joblib.dump(sgd, output_dir / prefixed(prefix, "sgd_regressor_model.pkl"))

    print("=== BERKA REGRESSION DONE ===")
    print(f"Ridge MAE: {ridge_metrics['mae']:.4f}, baseline persistence MAE: {base_last_metrics['mae']:.4f}")


if __name__ == "__main__":
    main()

