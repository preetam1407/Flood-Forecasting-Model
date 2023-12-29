import warnings
import json
import dataclasses
import pandas as pd
from datetime import datetime
import numpy as np
import re
import os
import rasterio
from rasterio import Affine, MemoryFile
from typing import Optional, Sequence, Tuple
from absl import logging
import pickle





filtered_df=pd.read_csv("Train_GaugeMeasurement_Final.csv")
# filtered_df=pd.read_csv("Train_GaugeMeasurement.csv")

@dataclasses.dataclass
class GroundTruthMeasurement:
    ground_truth: np.ndarray
    gauge_measurement: float
    date: str  

def load_inundation_maps(file_list):
    inundation_maps = []
    base_path = 'June_Sep_2001_2012_binary'
    for file in file_list:
        if file.endswith(".tif"):
            file_path = os.path.join(base_path, file)  # Correct file path
            with rasterio.open(file_path) as src:
                inundation_map = src.read(1)
            inundation_maps.append(inundation_map)
    return inundation_maps

def load_gauge_measurements(input_data):
    return input_data.set_index("filename")["Mean_Gauge"].to_dict()

filename_array=filtered_df['filename']
file_list = filename_array.tolist()

inundation_maps = load_inundation_maps(file_list)

# Load gauge measurements from the CSV file
gauge_measurements = load_gauge_measurements(filtered_df)

GROUND_TRUTH = [
    GroundTruthMeasurement(
        ground_truth=inundation_map,
        gauge_measurement=gauge_measurements[filename],
        date=re.search(r'\((\d{2}[A-Z]{3}\d{4})', filename).group(1)
    )
    for filename, inundation_map in zip(file_list, inundation_maps)
]




# Range above heighest measurement and below lowest measurement in which we expect to get measurements.
_MEASUREMENT_BUFFER_METERS = 10

# A number that is smaller than the uncertainty bound of a gauge measurement.
_EPSILON = 0.002


def get_cutoff_points(
    measurements: np.ndarray,
    precision_oriented: bool) -> Tuple[Sequence[int], np.ndarray]:

    # `measurements` is sorted in ascending order. Let's just verify this.
    if np.any(np.diff(measurements) < 0):
        raise ValueError('Expected measurements to be sorted: %r' % measurements)

    # Add an artificially low threshold value, to be used for pixels which should
    # always be wet.
    threshold_values = [measurements[0] - _MEASUREMENT_BUFFER_METERS]
    cutoff_points = [0]

    differing_measurements = ~np.isclose(measurements[:-1], measurements[1:])
    differing_locations = np.nonzero(differing_measurements)[0]
    cutoff_points.extend(differing_locations + 1)
    # For cutoffs higher than the lowest event and lower than the highest event,
    # any threshold value between measurements[i] and measurements[i+1] can be
    # used.
    if precision_oriented:
        # To be as conservative as possible, we choose measurements[i+1] - _EPSILON,
        # in order to include as small a region as possible in the risk map.
        threshold_values.extend(measurements[differing_locations + 1] - _EPSILON)
    else:
        # To alert as much as possible, we choose measurements[i],
        # in order to include as large a region as possible in the risk map.
        threshold_values.extend(measurements[differing_locations])

    # Add the final threshold. When not precision oriented, above the highest
    # event all pixels should be considered inundated. When precision oriented,
    # above the inundation map should be equal to that of the highest event.
    threshold_values.append(measurements[-1])
    if precision_oriented:
        threshold_values[-1] += _MEASUREMENT_BUFFER_METERS
        cutoff_points.append(len(measurements))

        logging.info('Found %d threshold values: %r', len(threshold_values),
                    np.asarray(threshold_values))

        # return cutoff_points, np.asarray(threshold_values, dtype=np.float)
        return cutoff_points, np.asarray(threshold_values, dtype=float)



def _count_true_in_suffixes(imaps: np.ndarray,
                            cutoff_points: Sequence[int]) -> np.ndarray:

    all_count_true = np.nancumsum(imaps[::-1], axis=0)[::-1]
    # If the suffix is empty, the number of True's is zero.
    filler_zeros = np.expand_dims(np.zeros_like(all_count_true[0]), 0)
    all_count_true = np.concatenate([all_count_true, filler_zeros])
    return all_count_true.take(cutoff_points, axis=0, mode='clip')


def count_true_wets_per_cutoff(imaps: np.ndarray,
                            cutoff_points: Sequence[int]) -> np.ndarray:

    return _count_true_in_suffixes(imaps, cutoff_points)


def count_false_wets_per_cutoff(imaps: np.ndarray,
                                cutoff_points: Sequence[int]) -> np.ndarray:

    not_imaps = 1 - imaps
    return _count_true_in_suffixes(not_imaps, cutoff_points)


def count_false_drys_per_cutoff(imaps: np.ndarray,
                                cutoff_points: Sequence[int]) -> np.ndarray:

    all_count_false_drys = np.nancumsum(imaps, axis=0)
    # If the cutoff is lower than all events, the number of false drys is zero.
    filler_zeros = np.expand_dims(np.zeros_like(all_count_false_drys[0]), 0)
    all_count_false_drys = np.concatenate([filler_zeros, all_count_false_drys])
    return all_count_false_drys.take(cutoff_points, axis=0, mode='clip')


def _get_pixel_threshold_index(pixel_events: np.ndarray,
                            cutoff_points: np.ndarray,
                            min_ratio: float) -> int:

    while np.nansum(pixel_events):
        true_wets = count_true_wets_per_cutoff(pixel_events, cutoff_points)
        false_wets = count_false_wets_per_cutoff(pixel_events, cutoff_points)
        ratios = true_wets / false_wets
        # The empty slice, corresponding to ratios[-1], has 0 true wets and 0 false
        # wets. We define the ratio there to be 0, as we want any slice that
        # contain a true wet to have a higher ratio than the empty slice.
        ratios[-1] = 0
        # Take the last maximum. In the case we have d, nan, w, we want the
        # threshold to be between nan and w, despite the fact that the threshold
        # between d and nan has the same ratio.
        best_index = ratios.shape[0] - 1 - np.nanargmax(ratios[::-1])

        if ratios[best_index] < min_ratio:
            break
        # Find the next candidate threshold on the prefix of all events below the
        # current best cutoff point.
        best_cutoff_point = cutoff_points[best_index]
        pixel_events = pixel_events[:best_cutoff_point]
        cutoff_points = cutoff_points[:best_index + 1]
    # If the remaining slice contains only NaNs, pixel should always be
    # inundated.
    if np.all(np.isnan(pixel_events)):
        return 0
    return len(cutoff_points) - 1


def _get_threshold_indices(imaps: np.ndarray, cutoff_points: np.ndarray,
                        min_ratio: float) -> np.ndarray:

    threshold_indices = np.zeros_like(imaps[0, :, :], dtype=int)
    for idx_y in range(threshold_indices.shape[0]):

        for idx_x in range(threshold_indices.shape[1]):
            threshold_indices[idx_y, idx_x] = _get_pixel_threshold_index(
            imaps[:, idx_y, idx_x], cutoff_points, min_ratio)
    return threshold_indices


def _learn_optimal_sar_prediction_internal(imaps: np.ndarray,
                                        measurements: np.ndarray,
                                        min_ratio: float) -> np.ndarray:
    cutoff_points, threshold_values = get_cutoff_points(
        measurements, precision_oriented=True)

    threshold_indices = _get_threshold_indices(imaps, np.array(cutoff_points),
                                                min_ratio)
    thresholds = threshold_values[threshold_indices]

    if np.issubdtype(imaps.dtype, np.floating):
        # If all inundation maps are NaN for a given pixel, make the threshold for
        # that pixel a NaN as well. This occurs when the pixel is outside the
        # requested forecast_region.
        thresholds = np.where(np.all(np.isnan(imaps), axis=0), np.nan, thresholds)

    return thresholds


def masked_array_to_float_array(masked_array):
    float_array = np.array(masked_array, dtype=float)
    mask = np.ma.getmaskarray(masked_array)
    float_array[mask] = np.nan
    return float_array


def learn_optimal_sar_prediction_from_ground_truth(
    flood_events_train: Sequence[GroundTruthMeasurement],
    min_ratio: Optional[float] = None) -> np.ndarray:

    # Sort the gauge measurements in ascending order
    measurements = np.asarray(sorted([fe.gauge_measurement for fe in flood_events_train]))

    # Rest of your code remains unchanged
    imaps = np.asarray([
        masked_array_to_float_array(fe.ground_truth) for fe in flood_events_train
    ])

    return _learn_optimal_sar_prediction_internal(imaps, measurements, min_ratio)


MIN_RATIOS = [1]

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





tm = ThresholdingModel()
tm.train(GROUND_TRUTH)


predicted_result=tm.infer(12)
print(predicted_result)



# Save the trained model to a pickle file
with open('thresholding_model.pkl', 'wb') as model_file:
    pickle.dump(tm, model_file)

