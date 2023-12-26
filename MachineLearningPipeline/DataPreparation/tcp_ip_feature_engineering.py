from typing import Optional

from utils import *
import numpy as np
import math
from scipy import stats
from statsmodels.tsa import stattools as st
from statsmodels.tsa.stattools import adfuller
from collections import Counter


class TcpIpFeatureProcessor:
    def __init__(self, df):
        self.df = df

    @staticmethod
    def calculate_burstiness(features_list: list) -> tuple:
        """
        Calculate the burstiness of a list of features
        :param features_list: List of Ints/Floats
        :return: Float indicating the bustiness calculated
        [cv: Coefficient of variance, bp: Burstiness Parameter,id: Index of Dispersion]
        """
        mean = np.mean(features_list)
        std = np.std(features_list)
        var = np.var(features_list)
        if (mean == 0) | (std == 0):
            return 0, 0, 0
        else:
            return std / mean, (std - mean) / (std + mean), var / mean

    @staticmethod
    def detect_epochtime_burstiness(epochtimesubs: list, burst_threshold=0.05, idle_threshold=0.1) -> tuple:
        """
        Calculates given an idle and burst threshold the count of bursts, idle time and bloc of session
        :param epochtimesubs: List containing the subs of epochtime
        :param burst_threshold: threshold of the burst
        :param idle_threshold: threshold of the idle
        :return: current_session, idle_time, burst_counter
        """
        current_burst = 0
        current_session = 0
        burst_counter = 0  # total bursts encountered
        session_counter = 0  # total packets containing bursts
        idle_time = 0

        for e in epochtimesubs:
            if e < burst_threshold:  # if the current value is below threshold
                current_burst += 1  # part of the current burst
                current_session += 1  # part of the current packet
            else:  # new burst
                burst_counter += 1  # capture bursts
                current_burst = 1  # reset to 1
                if e > idle_threshold:  # if the epoch time value is above the idle_threshold
                    current_session = 1  # new session
                    session_counter += 1
                    idle_time += e
        return current_session, idle_time, burst_counter

    @staticmethod
    def calculate_feature_difference(val1: list, val2: list, op='max') -> float:
        """
        Calculate the difference between the forward and backward packet
        :param val1: 1st list
        :param val2: 2nd list
        :param op: [max:using the maximum of packet, sum: using the total of the packet]
        :return: Float
        """
        if op == 'max':
            return abs(max(val1) - max(val2))
        elif op == 'sum':
            return abs(sum(val1) - sum(val2))
        else:
            raise Exception("Invalid Operation")

    @staticmethod
    def calculate_ratio(features_list: list, method: str) -> float:
        """
        Calculate the ratio of a list of features
        :param features_list: List of Ints/Floats
        :param method: [max:divide mean of the list by the max of the list, total:divide the maximum of the list
        by the total]
        :return: Float
        """
        if method not in ['max', 'total']:
            raise ValueError("Invalid method", method)
        else:
            max_value = max(features_list)
            if method == 'max':
                mean_value = np.mean(features_list)
                return mean_value / max_value if (max_value != 0) else 0

            elif method == 'total':
                sum_value = sum(features_list)
                return max_value / sum_value if (sum_value != 0) else 0

    @staticmethod
    def calculate_ratio_forward_backward(features_list1: list, features_list2: list) -> float:
        """
        Calculate the ratio of forward by backward packets
        :param features_list1: 1st list containing the forward packets
        :param features_list2: 2nd list containing the backward packets
        :return: Float
        """
        if np.mean(features_list2) == 0:
            return 0
        else:
            return np.mean(features_list1) / np.mean(features_list2)

    @staticmethod
    def compute_stationarity(time_serie_val: list) -> bool:
        """
        Compute stationarity given a list of time series values
        :param time_serie_val: list, time seris list
        :return: Boolean
        """
        try:
            nb_unique = len(set(time_serie_val))
            if nb_unique > 1:
                # Calculate the ADF statistic and p-value using the adfuller function
                adf_result = adfuller(time_serie_val)
                p_value = adf_result[1]
                # Create a new feature called 'stationarity' that indicates whether the data is stationary or not
                stationarity = True if p_value < 0.05 else False
            else:
                stationarity = True

        except Exception as e:
            stationarity = True

        return stationarity

    @staticmethod
    def compute_time_series_characteristics(series: list) -> tuple:
        """
        Computes the trend, periodicity, and seasonality of a given time series.
        :param series: A list or numpy array containing the time series data
        :return:
            trend : Trend of the time serie list computed as the slope of the line that best fits the time series data
                        using linear regression,
            periodicity : Te number of time steps between two consecutive peaks (if any) in the autocorrelation function
             of the residuals. If the time series is constant or has no significant autocorrelation,
             the periodicity is set to 0.
            seasonal : Whether the time series exhibits seasonality or not. A time series is considered seasonal
            if the autocorrelation function of the residuals shows at least one peak that is statistically significant
             at the 5% level. If the time series is constant or has no significant autocorrelation,
              the seasonal flag is set to False.
        """
        if len(series) == 0 or len(set(series)) <= 1:
            return False, 0, 0
        nb_unique = len(set(series))
        seasonal = False  # false if length of list=0 or periodicity ==1
        trend = 0
        periodicity = 0
        if nb_unique > 1:
            # Calculate the trend in the data
            trend = np.polyfit(range(len(series)), series, 1)[0]
            # Calculate the residuals in the data
            residuals = series - (trend * np.array(range(len(series))))
            # Test for seasonality
            serie = pd.Series(series)
            # Calculate the autocorrelation with a lag of 1
            autocorr = serie.autocorr()
            if autocorr != 0 and not math.isnan(autocorr):
                periodicity = round(1 / autocorr)
                if periodicity > 1:
                    acf, q, p = st.acf(residuals, nlags=periodicity, qstat=True, fft=False)
                    # Check if residuals have seasonality
                    seasonal = True if (p < 0.05).any() else False
                else:
                    periodicity = len(series)
            else:
                periodicity = 0
        return trend, periodicity, seasonal

    @staticmethod
    def entropy(list_values: list) -> Optional[float]:
        """
        The function calculates the entropy of a list of values
        :param list_values: list of values
        :return: Float or None if list is empty
        """
        if not list_values:
            return None
        counter = Counter(list_values)
        total = len(list_values)
        entropy = 0
        log2 = math.log2
        for count in counter.values():
            probability = count / total
            entropy -= probability * log2(probability)
        return entropy

    @staticmethod
    def calculate_diffs(features_list: list) -> list:
        """
        Takes a list of integer or float values and generates a new list as the difference between
        two successive values in the initial list.
        :param features_list: A list of integers or floats.
        :return: A list of floats or integers representing the differences between the successive
             elements in the input list. Returns an empty list if the input list contains less
             than two valid elements or if any element is not an integer or a float.
        """
        if len(features_list) > 1:
            diff = np.diff(features_list).tolist()
        else:
            diff = [0]
        return diff

    def process_flags(self, direction: str):
        """
        Count the occurrence of each flag
        :param direction: [I:Backward, O:Forward]
        :return: Dataframe
        """
        unique_values = pd.Series(
            self.df['new_tcp.flags_20_False_' + direction].map(list_of_list_to_list).explode().unique())
        flag_counts = self.df['new_tcp.flags_20_False_' + direction].map(list_of_list_to_list).apply(
            lambda x: pd.Series([x.count(v) for v in unique_values]))
        flag_counts.columns = ['new_tcp.flags_'] + unique_values + ['_count_' + direction]
        self.df = pd.concat([self.df, flag_counts], axis=1)

    def calculate_additional_features(self):
        self.df['new_tcp.window_size_value_diff'] = self.df.apply(
            lambda x: self.calculate_feature_difference(x['new_tcp.window_size_value_20_False_I'],
                                                        x['new_tcp.window_size_value_20_False_O'], 'max'), axis=1)

        self.df['new_tcp.window_size_value_ratio_forward_backward'] = self.df.apply(
            lambda x: self.calculate_ratio_forward_backward(x['new_tcp.window_size_value_20_False_O'],
                                                            x['new_tcp.window_size_value_20_False_I']), axis=1)

        self.df['new_ip.len_diff'] = self.df.apply(
            lambda x: self.calculate_feature_difference(x['new_ip.len_20_False_I'], x['new_ip.len_20_False_O'], 'sum'),
            axis=1)

        self.df['new_ip.len_ratio_forward_backward'] = self.df.apply(
            lambda x: self.calculate_ratio_forward_backward(x['new_ip.len_20_False_O'], x['new_ip.len_20_False_I']),
            axis=1)

    def compute_statistical_features(self, row: str, config: dict) -> pd.Series:
        """
        This function calculate statistical features for each row based on list of the specified list of operations
        :param row: str, referencing name of the column
        :param config: dictionary of the list operations
        :return:  Series of the new features
        """
        common_operations = {
            'min': np.min, 'max': np.max, 'mean': np.mean, 'std': np.std, 'var': np.var, 'median': np.median,
            'Q1': lambda x: np.quantile(x, 0.25), 'Q3': lambda x: np.quantile(x, 0.75), 'IQR': stats.iqr,
            'perc_90': lambda x: np.percentile(x, 90), 'perc_95': lambda x: np.percentile(x, 95),
            'entropy': self.entropy, 'skew': stats.skew, 'kurt': stats.kurtosis}
        features = {}

        for feature, operations in config['tcpip_operations'].items():
            for i in ['I', 'O']:
                for op_name, op_func in common_operations.items():
                    features[f'{feature}_{i}_{op_name}'] = op_func(row[f'{feature}_{i}'])
                if 'rate_max' in operations[i]:
                    features[f'{feature}_{i}_rate_max'] = self.calculate_ratio(row[f'{feature}_{i}'], 'max')
                if 'rate_total' in operations[i]:
                    features[f'{feature}_{i}_rate_total'] = self.calculate_ratio(row[f'{feature}_{i}'], 'total')
                if 'total' in operations[i]:
                    features[f'{feature}_{i}_total'] = np.sum(row[f'{feature}_{i}'])
                if 'total_time' in operations[i]:
                    features[f'{feature}_{i}_total_time'] = row[f'{feature}_{i}'][-1] - row[f'{feature}_{i}'][0]
                if 'cv' in operations or 'id' in operations or 'bp' in operations:
                    burstiness = self.calculate_burstiness(row[f'{feature}_{i}'])
                    if 'cv' in operations:
                        features[f'{feature}_{i}_cv'] = burstiness[0]
                    if 'id' in operations:
                        features[f'{feature}_{i}_id'] = burstiness[2]
                    if 'bp' in operations:
                        features[f'{feature}_{i}_bp'] = burstiness[1]
                if 'total_burst' in operations[i] or 'idle_time' in operations[i] or 'bursty_sessions' in operations[i]:
                    burstiness = self.detect_epochtime_burstiness(row[f'{feature}_{i}'])
                    if 'total_burst' in operations[i]:
                        features[f'{feature}_{i}_total_burst'] = burstiness[0]
                    if 'idle_time' in operations[i]:
                        features[f'{feature}_{i}_idle_time'] = burstiness[1]
                    if 'bursty_sessions' in operations[i]:
                        features[f'{feature}_{i}_bursty_sessions'] = burstiness[2]
                if 'trend' in operations[i] or 'periodicity' in operations[i] or 'seasonal' in operations[i]:
                    ts = self.compute_time_series_characteristics(row[f'{feature}_{i}'])
                    if 'trend' in operations[i]:
                        features[f'{feature}_{i}_trend'] = ts[0]
                    if 'periodicity' in operations[i]:
                        features[f'{feature}_{i}_periodicity'] = ts[1]
                    if 'seasonal' in operations[i]:
                        features[f'{feature}_{i}_seasonal'] = ts[2]
                if 'stationarity' in operations[i]:
                    features[f'{feature}_{i}_stationarity'] = self.compute_stationarity(row[f'{feature}_{i}'])
        return pd.Series(features)

    def preprocess_tcpip_features(self, config: dict) -> pd.DataFrame:
        """
        Process the TCP/IP features
        :param config: Config file
        :return: Dataframe
        """

        for i in ['I', 'O']:
            self.process_flags(i)
            self.df['new_frame.time_epochsub_20_False_' + i] = self.df['new_frame.time_epoch_20_False_' + i].apply(
                self.calculate_diffs)

        statistical_features = self.df.apply(self.compute_statistical_features, args=(config,), axis=1)
        self.df = pd.concat([self.df, statistical_features], axis=1)
        self.calculate_additional_features()

        return self.df
