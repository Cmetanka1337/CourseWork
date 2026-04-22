import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class CalibratorConfig:
    n_classes: int = 4
    learning_rate: float = 0.05
    l2: float = 1e-3
    grad_clip_norm: float = 5.0
    seed: int = 42


class SoftmaxCalibrator:
    """On-device friendly multiclass calibrator over RF probabilities.

    Formula: p_adj = softmax(W @ p + b), where p is RF probability vector.
    """

    def __init__(
        self,
        n_classes: int = 4,
        learning_rate: float = 0.05,
        l2: float = 1e-3,
        grad_clip_norm: float = 5.0,
        seed: int = 42,
    ) -> None:
        self.config = CalibratorConfig(
            n_classes=n_classes,
            learning_rate=learning_rate,
            l2=l2,
            grad_clip_norm=grad_clip_norm,
            seed=seed,
        )
        self.rng = np.random.default_rng(seed)
        self.W = np.eye(n_classes, dtype=float)
        self.b = np.zeros(n_classes, dtype=float)

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exps = np.exp(shifted)
        return exps / np.sum(exps, axis=1, keepdims=True)

    def predict_proba(self, p_rf: np.ndarray) -> np.ndarray:
        x = np.asarray(p_rf, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.shape[1] != self.config.n_classes:
            raise ValueError(f"Expected input with {self.config.n_classes} classes, got {x.shape[1]}")
        logits = x @ self.W.T + self.b
        return self._softmax(logits)

    def train_batch(self, p_rf: np.ndarray, y_true: np.ndarray, epochs: int = 1) -> dict:
        x = np.asarray(p_rf, dtype=float)
        y = np.asarray(y_true, dtype=int)

        if x.ndim != 2:
            raise ValueError("p_rf must be 2D array")
        if x.shape[1] != self.config.n_classes:
            raise ValueError(f"Expected {self.config.n_classes} columns in p_rf, got {x.shape[1]}")
        if y.ndim != 1 or len(y) != len(x):
            raise ValueError("y_true shape mismatch")

        n = max(len(x), 1)
        one_hot = np.zeros((len(x), self.config.n_classes), dtype=float)
        one_hot[np.arange(len(x)), y] = 1.0

        last_loss = 0.0
        for _ in range(max(1, int(epochs))):
            probs = self.predict_proba(x)
            eps = 1e-12
            last_loss = float(-np.sum(one_hot * np.log(np.clip(probs, eps, 1.0))) / n)

            dlogits = (probs - one_hot) / n
            grad_W = dlogits.T @ x + self.config.l2 * self.W
            grad_b = dlogits.sum(axis=0)

            grad_norm = float(np.sqrt(np.sum(grad_W**2) + np.sum(grad_b**2)))
            if grad_norm > self.config.grad_clip_norm:
                scale = self.config.grad_clip_norm / max(grad_norm, 1e-12)
                grad_W *= scale
                grad_b *= scale

            self.W -= self.config.learning_rate * grad_W
            self.b -= self.config.learning_rate * grad_b

        return {"loss": last_loss}

    def to_dict(self) -> dict:
        return {
            "config": {
                "n_classes": self.config.n_classes,
                "learning_rate": self.config.learning_rate,
                "l2": self.config.l2,
                "grad_clip_norm": self.config.grad_clip_norm,
                "seed": self.config.seed,
            },
            "W": self.W.tolist(),
            "b": self.b.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "SoftmaxCalibrator":
        cfg = payload["config"]
        obj = cls(
            n_classes=int(cfg["n_classes"]),
            learning_rate=float(cfg["learning_rate"]),
            l2=float(cfg["l2"]),
            grad_clip_norm=float(cfg["grad_clip_norm"]),
            seed=int(cfg.get("seed", 42)),
        )
        obj.W = np.asarray(payload["W"], dtype=float)
        obj.b = np.asarray(payload["b"], dtype=float)
        return obj

    def save_json(self, path: Path) -> None:
        out = Path(path).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load_json(cls, path: Path) -> "SoftmaxCalibrator":
        payload = json.loads(Path(path).resolve().read_text(encoding="utf-8"))
        return cls.from_dict(payload)

