import dataclasses
import numpy as np
from typing import Optional, Sequence, Tuple



@dataclasses.dataclass
class GroundTruthMeasurement:
    ground_truth: np.ndarray
    gauge_measurement: float
    date: str 
        

class ThresholdingModel:
    def f1_metric(self, ground_truths: Sequence[GroundTruthMeasurement],
                thresholds: np.ndarray) -> float:
        """Returns the aggregated F1 metric for a set of thresholds."""
        relevant_true = predicted_true = true_positives = 0
        for gt in ground_truths:
            predicted = thresholds < gt.gauge_measurement
            actual = gt.ground_truth.astype(bool)
            true_positives += np.sum(predicted & actual)
            predicted_true += np.sum(predicted)
            relevant_true += np.sum(actual)

        total_precision = true_positives / predicted_true
        total_recall = true_positives / relevant_true
        return 2/(1/total_precision + 1/total_recall)

    def train(self, ground_truth: Sequence[GroundTruthMeasurement]):
        """Trains the model according to the provided training set."""
        min_ratio_to_thresholds = {}
        f1_to_min_ratio = {}
        for min_ratio in MIN_RATIOS:
            min_ratio_to_thresholds[min_ratio] = (
                learn_optimal_sar_prediction_from_ground_truth(GROUND_TRUTH, 
                                                            min_ratio))
            f1 = self.f1_metric(ground_truth, 
                                min_ratio_to_thresholds[min_ratio])
            print(f'For min_ratio={min_ratio} we get f1={f1}')
            f1_to_min_ratio[f1] = min_ratio

        best_f1 = max(f1_to_min_ratio.keys())
        best_min_ratio = f1_to_min_ratio[best_f1]
        print('chosen min_ratio', best_min_ratio)
        self.thresholds = min_ratio_to_thresholds[best_min_ratio]

    # def infer(self, gauge_level: float):
    #     """Returns the inferred inundation model for a gauge level."""
    #     return self.thresholds < gauge_level

    def infer(self, gauge_level: float):
        """Returns the inferred inundation model for a gauge level."""
        gauge_level = np.float64(gauge_level)  # Ensure the same data type
        return self.thresholds < gauge_level
    