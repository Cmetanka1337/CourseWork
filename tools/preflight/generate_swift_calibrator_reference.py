#!/usr/bin/env python3
import sys
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> int:
    root = repo_root()
    bundle_dir = root / "artifacts/ios_bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    swift_path = bundle_dir / "SoftmaxCalibratorReference.swift"
    swift_path.write_text(
        """import Foundation

struct CalibratorState: Codable {
    var W: [[Double]]
    var b: [Double]
}

struct SoftmaxCalibratorReference {
    static func softmax(_ logits: [Double]) -> [Double] {
        let maxLogit = logits.max() ?? 0.0
        let expValues = logits.map { Foundation.exp($0 - maxLogit) }
        let denom = expValues.reduce(0.0, +)
        return expValues.map { $0 / denom }
    }

    static func forward(pRF: [Double], W: [[Double]], b: [Double]) -> [Double] {
        var logits = Array(repeating: 0.0, count: b.count)
        for i in 0..<W.count {
            var acc = b[i]
            for j in 0..<pRF.count {
                acc += W[i][j] * pRF[j]
            }
            logits[i] = acc
        }
        return softmax(logits)
    }

    static func sgdUpdate(
        pRF: [Double],
        yTrue: Int,
        W: [[Double]],
        b: [Double],
        lr: Double,
        l2: Double,
        clip: Double
    ) -> ([[Double]], [Double]) {
        let logits = forwardLogits(pRF: pRF, W: W, b: b)
        let pAdj = softmax(logits)

        var gradLogits = pAdj
        if yTrue >= 0 && yTrue < gradLogits.count {
            gradLogits[yTrue] -= 1.0
        }

        var gradW = W
        var gradB = gradLogits
        for i in 0..<W.count {
            for j in 0..<W[i].count {
                gradW[i][j] = gradLogits[i] * pRF[j] + l2 * W[i][j]
            }
        }

        var norm = 0.0
        for i in 0..<gradW.count {
            for j in 0..<gradW[i].count {
                norm += gradW[i][j] * gradW[i][j]
            }
        }
        for value in gradB {
            norm += value * value
        }
        norm = Foundation.sqrt(norm)

        let scale = norm > clip && norm > 0.0 ? (clip / norm) : 1.0
        if scale != 1.0 {
            for i in 0..<gradW.count {
                for j in 0..<gradW[i].count {
                    gradW[i][j] *= scale
                }
            }
            for i in 0..<gradB.count {
                gradB[i] *= scale
            }
        }

        var nextW = W
        var nextB = b
        for i in 0..<W.count {
            for j in 0..<W[i].count {
                nextW[i][j] = W[i][j] - lr * gradW[i][j]
            }
        }
        for i in 0..<b.count {
            nextB[i] = b[i] - lr * gradB[i]
        }

        return (nextW, nextB)
    }

    static func encodeState(_ state: CalibratorState) throws -> String {
        let data = try JSONEncoder().encode(state)
        return String(data: data, encoding: .utf8) ?? ""
    }

    private static func forwardLogits(pRF: [Double], W: [[Double]], b: [Double]) -> [Double] {
        var logits = Array(repeating: 0.0, count: b.count)
        for i in 0..<W.count {
            var acc = b[i]
            for j in 0..<pRF.count {
                acc += W[i][j] * pRF[j]
            }
            logits[i] = acc
        }
        return logits
    }
}

// JSON shape: {"W": [[Double]], "b": [Double]}
""",
        encoding="utf-8",
    )

    print(f"Swift reference written to: {swift_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

