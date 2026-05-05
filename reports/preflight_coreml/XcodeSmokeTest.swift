import CoreML
import Foundation

struct GoldenExample {
    let expectedClass: Int
    let features: [String: Double]
}

// Option A (preferred if generated interface exists):
// let model = try BerkaSpendBucketRF(configuration: MLModelConfiguration())
// let output = try model.prediction(bucket_spend_t: ..., weeks_since_outflow: ...)

func loadModel() throws -> MLModel {
    if let url = Bundle.main.url(forResource: "BerkaSpendBucketRF", withExtension: "mlpackage") {
        return try MLModel(contentsOf: url)
    }
    let fallback = URL(fileURLWithPath: "artifacts/coreml/BerkaSpendBucketRF.mlpackage")
    return try MLModel(contentsOf: fallback)
}

func extractClassLabel(_ output: MLFeatureProvider) -> Int? {
    for name in output.featureNames {
        guard let value = output.featureValue(for: name) else { continue }
        if value.type == .int64 { return Int(value.int64Value) }
        if value.type == .string, let parsed = Int(value.stringValue) { return parsed }
    }
    return nil
}

func extractProbabilities(_ output: MLFeatureProvider) -> [String: Double]? {
    for name in output.featureNames {
        guard let value = output.featureValue(for: name) else { continue }
        if value.type == .dictionary {
            let dict = value.dictionaryValue
            var result: [String: Double] = [:]
            for (key, val) in dict {
                result[String(describing: key)] = val.doubleValue
            }
            return result
        }
    }
    return nil
}

let examples: [GoldenExample] = [
    GoldenExample(
        expectedClass: 0,
        features: [
            "bucket_spend_t": 3.0,
            "bucket_net_t": 3.0,
            "weekly_inflow_t": 23752.0,
            "weekly_outflow_t": 17084.0,
            "weekly_net_t": 6668.0,
            "txn_count_t": 3.0,
            "category_diversity_t": 3.0,
            "weekly_inflow_t_minus_1": 162.9,
            "weekly_outflow_t_minus_1": 714.6,
            "weekly_net_t_minus_1": -551.7,
            "weekly_inflow_t_minus_2": 0.0,
            "weekly_outflow_t_minus_2": 1300.0,
            "outflow_inflow_ratio_t": 0.7192657460424385,
            "week_of_year": 50.0,
            "month": 12.0,
            "quarter": 4.0,
            "week_of_month": 1.0,
            "is_month_start_week": 1.0,
            "is_month_end_week": 0.0,
            "delta_inflow": 23589.1,
            "delta_outflow": 16369.4,
            "inflow_outflow_ratio": 1.3903067196563856,
            "inflow_share": 0.5816436477475354,
            "inflow_rolling_mean_8w": 6000.4,
            "inflow_rolling_std_8w": 8144.113760256541,
            "outflow_rolling_mean_8w": 5986.974999999999,
            "outflow_rolling_std_8w": 7042.181990132237,
            "inflow_frequency_8w": 0.75,
            "outflow_frequency_8w": 0.875,
            "weeks_since_inflow": 2.0,
            "weeks_since_outflow": 1.0
        ]
    ),
    GoldenExample(
        expectedClass: 3,
        features: [
            "bucket_spend_t": 1.0,
            "bucket_net_t": 2.0,
            "weekly_inflow_t": 52.8,
            "weekly_outflow_t": 14.6,
            "weekly_net_t": 38.2,
            "txn_count_t": 2.0,
            "category_diversity_t": 2.0,
            "weekly_inflow_t_minus_1": 0.0,
            "weekly_outflow_t_minus_1": 11816.0,
            "weekly_net_t_minus_1": -11816.0,
            "weekly_inflow_t_minus_2": 23122.0,
            "weekly_outflow_t_minus_2": 11700.0,
            "outflow_inflow_ratio_t": 0.2765151515151515,
            "week_of_year": 44.0,
            "month": 10.0,
            "quarter": 4.0,
            "week_of_month": 4.0,
            "is_month_start_week": 0.0,
            "is_month_end_week": 0.0,
            "delta_inflow": 52.8,
            "delta_outflow": -11801.4,
            "inflow_outflow_ratio": 3.6164381084631434,
            "inflow_share": 0.7833827776946176,
            "inflow_rolling_mean_8w": 8681.4375,
            "inflow_rolling_std_8w": 11967.765271993583,
            "outflow_rolling_mean_8w": 7934.650000000001,
            "outflow_rolling_std_8w": 4965.558943361758,
            "inflow_frequency_8w": 0.5,
            "outflow_frequency_8w": 1.0,
            "weeks_since_inflow": 1.0,
            "weeks_since_outflow": 1.0
        ]
    ),
    GoldenExample(
        expectedClass: 1,
        features: [
            "bucket_spend_t": 2.0,
            "bucket_net_t": 2.0,
            "weekly_inflow_t": 2966.0,
            "weekly_outflow_t": 523.0,
            "weekly_net_t": 2443.0,
            "txn_count_t": 2.0,
            "category_diversity_t": 2.0,
            "weekly_inflow_t_minus_1": 0.0,
            "weekly_outflow_t_minus_1": 1814.0,
            "weekly_net_t_minus_1": -1814.0,
            "weekly_inflow_t_minus_2": 68.5,
            "weekly_outflow_t_minus_2": 14.6,
            "outflow_inflow_ratio_t": 0.1763317599460553,
            "week_of_year": 46.0,
            "month": 11.0,
            "quarter": 4.0,
            "week_of_month": 2.0,
            "is_month_start_week": 0.0,
            "is_month_end_week": 0.0,
            "delta_inflow": 2966.0,
            "delta_outflow": -1291.0,
            "inflow_outflow_ratio": 5.671128096231112,
            "inflow_share": 0.850100315032932,
            "inflow_rolling_mean_8w": 1136.8125,
            "inflow_rolling_std_8w": 1514.9547324095197,
            "outflow_rolling_mean_8w": 994.35,
            "outflow_rolling_std_8w": 812.6488522989848,
            "inflow_frequency_8w": 0.75,
            "outflow_frequency_8w": 1.0,
            "weeks_since_inflow": 1.0,
            "weeks_since_outflow": 1.0
        ]
    ),
]

do {
    let model = try loadModel()
    for (idx, example) in examples.enumerated() {
        let provider = try MLDictionaryFeatureProvider(dictionary: example.features)
        let start = CFAbsoluteTimeGetCurrent()
        let prediction = try model.prediction(from: provider)
        let durationMs = (CFAbsoluteTimeGetCurrent() - start) * 1000.0
        let predictedClass = extractClassLabel(prediction)
        let probs = extractProbabilities(prediction)
        print("Row \(idx): expected=\(example.expectedClass) predicted=\(predictedClass ?? -1) latency_ms=\(durationMs)")
        if let probs = probs {
            print("probs=\(probs)")
        }
    }
} catch {
    print("Smoke test failed: \(error)")
}
