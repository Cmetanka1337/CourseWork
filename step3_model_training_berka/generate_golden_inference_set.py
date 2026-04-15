import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from train_classification import CLASSIFICATION_FEATURES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate golden inference set with predict_proba")
    parser.add_argument(
        "--test-csv",
        type=str,
        default="step1_berka_weekly_builder/outputs/classification/test_lag_features.csv",
        help="Path to classification test_lag_features.csv",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="step3_model_training_berka/outputs/full_spend_tuned_rf_model.pkl",
        help="Path to tuned RF model",
    )
    parser.add_argument("--target", type=str, default="bucket_spend_t_plus_1")
    parser.add_argument("--sample-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-csv",
        type=str,
        default="step3_model_training_berka/outputs/golden_inference_set_full_spend_tuned.csv",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="step3_model_training_berka/outputs/golden_inference_set_full_spend_tuned.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    test_csv = Path(args.test_csv).resolve()
    model_path = Path(args.model_path).resolve()
    out_csv = Path(args.output_csv).resolve()
    out_json = Path(args.output_json).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(test_csv)
    required_cols = ["user_id", "week_start", "week_t", args.target] + CLASSIFICATION_FEATURES
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns in test set: {missing}")

    if len(df) < args.sample_size:
        raise RuntimeError(f"Requested sample-size={args.sample_size}, but test set has only {len(df)} rows")

    sampled = df.sample(n=int(args.sample_size), random_state=int(args.seed)).copy()
    sampled = sampled.sort_index()
    sampled.insert(0, "test_row_index", sampled.index.astype(int))

    model = joblib.load(model_path)
    x_sample = sampled[CLASSIFICATION_FEATURES].copy()
    proba = model.predict_proba(x_sample)
    classes = [int(x) for x in model.classes_]

    proba_cols = [f"proba_class_{c}" for c in classes]
    proba_df = pd.DataFrame(proba, columns=proba_cols, index=sampled.index)
    predicted_idx = np.argmax(proba, axis=1)
    predicted_class = [classes[i] for i in predicted_idx]

    export_cols = ["test_row_index", "user_id", "week_start", "week_t", args.target] + CLASSIFICATION_FEATURES
    out_df = pd.concat([sampled[export_cols].reset_index(drop=True), proba_df.reset_index(drop=True)], axis=1)
    out_df["predicted_class"] = predicted_class

    out_df.to_csv(out_csv, index=False)

    payload = {
        "execution_timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": int(args.seed),
        "sample_size": int(args.sample_size),
        "target": args.target,
        "model_path": str(model_path),
        "test_csv": str(test_csv),
        "feature_order": CLASSIFICATION_FEATURES,
        "class_order": classes,
        "records": out_df.to_dict(orient="records"),
    }
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Golden inference set written:")
    print(out_csv)
    print(out_json)


if __name__ == "__main__":
    main()

