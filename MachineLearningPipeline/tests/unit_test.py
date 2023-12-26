import sys

import pandas as pd
import unittest

from DataPreparation.data_preparation import *
from DataPreparation.session_feature_extraction import PacketProcessor
from DataPreparation.tcp_ip_feature_engineering import TcpIpFeatureProcessor
from DataPreparation.tls_feature_engineering import TLSFeatureProcessor
from utils import *


class TestFunctions(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame({'ht': ["[[1], [], [2], [], [], [], [], [], []]",
                                       "[[1], [], [2], [], [], [], [], [], []]"],
                                'ext': ["[[47802, 0, 23, 65281], [], [43, 51], [], [], [], [5]]",
                                        "[[47802, 0, 15, 13], [], [55, 51], [], [], [], [5]]"],
                                'ext1': [['47802', '0', '23', '65281'], ['47802', '0', '15', '13']],
                                'ext13': [None, None]
                                })
        self.df1 = pd.DataFrame({'ht': [[[1], [], [2], [], [], [], [], [], []]],
                                 'ext': [[[47802, 0, 23, 65281], [], [43, 51], [], [], [], [5]]]
                                 })
        self.df3 = pd.DataFrame({
            'column_name': ['A', 'B', 'C'],
            'direction_column': ['O', 'I', 'O']
        })

    def test_find_index(self):
        # Test that find_ht returns the expected index when the value is found in the list
        self.assertEqual(find_index(self.df1.iloc[0], 'ht', 1), 0)
        self.assertEqual(find_index(self.df1.iloc[0], 'ht', 2), 2)
        # Test that find_ht returns -1 when the value is not found in the list
        self.assertEqual(find_index(self.df1.iloc[0], 'ht', 14), -1)

    def test_get_handshake_type_index(self):
        # Test that get_handshake_type_index returns a dataframe with the expected columns
        get_handshake_type_index(self.df1, [1, 2, 3], 'ht')

        self.assertTrue('ht_pos1' in self.df1.columns)
        self.assertTrue('ht_pos2' in self.df1.columns)
        self.assertTrue('ht_pos3' in self.df1.columns)

        self.assertEqual(list(self.df1['ht_pos1']), [0])
        self.assertEqual(list(self.df1['ht_pos2']), [2])
        self.assertEqual(list(self.df1['ht_pos3']), [-1])

    def test_get_description_extension(self):
        # Test that the function correctly splits the column into dummy columns
        self.df['list_values'] = ["['0x9a9a', '0x1301', '0xc02b', '0xc030']", "['0x9a9a', '0xc02b', '0xc030']"]
        self.dictionary = {
            '59': 'dnssec_chain',
            '60-2569': 'Unassigned',
        }
        result = get_description_extension('59', self.dictionary)
        result1 = get_description_extension('61', self.dictionary)
        result2 = get_description_extension('31', self.dictionary)
        self.assertEqual(result, 'dnssec_chain')
        self.assertEqual(result1, 'Unassigned')
        self.assertEqual(result2, 'No_Extension_mapped')

    def test_convert_str_list_to_list(self):
        value = "[abc, def, ghi]"
        self.assertRaises(SyntaxError, convert_str_list_to_list, value)

    def test_convert_str_list_to_list_int(self):
        value = "['1', '2', '3']"
        self.assertEqual([1, 2, 3], convert_str_list_to_list_int(value))

    def test_list_of_list_to_list(self):
        value = [['1'], ['2'], ['3']]
        self.assertEqual(['1', '2', '3'], list_of_list_to_list(value))

    def test_filter_packet_by_direction(self):
        pp = PacketProcessor(self.df3)
        pp.filter_packet_in_direction('column_name', 'direction_column', 'O')
        self.assertTrue('column_name_O' in self.df3.columns)

    def test_entropy_non_empty_list(self):
        tcp = TcpIpFeatureProcessor(self.df)
        result = tcp.entropy([1, 2, 3, 4, 5])
        self.assertAlmostEqual(result, 2.32193, places=5)

    def test_entropy_empty_list(self):
        tcp = TcpIpFeatureProcessor(self.df)
        result = tcp.entropy([])
        self.assertEqual(result, None)

    def test_entropy(self):
        list_values = [1, 2, 3, 4, 5, 6]
        expected_result = 2.585
        tcp = TcpIpFeatureProcessor(self.df)
        result = tcp.entropy(list_values)
        self.assertAlmostEqual(expected_result, result, places=3)

    def test_compute_stationarity(self):
        time_serie_val = [1, 2, 3, 4, 5]
        expected_result = False
        tcp = TcpIpFeatureProcessor(self.df)
        result = tcp.compute_stationarity(time_serie_val)
        self.assertEqual(expected_result, result)

    def test_compute_stationarity_one_unique_value(self):
        time_serie_val = [1, 1, 1, 1, 1]
        expected_result = True
        tcp = TcpIpFeatureProcessor(self.df)
        result = tcp.compute_stationarity(time_serie_val)
        self.assertEqual(expected_result, result)

    def test_calculate_burstiness(self):
        features_list = [1, 2, 3, 4, 5]
        expected_result = (0.47140452079103173, -0.3592455179659185, 0.6666666666666666)
        tcp = TcpIpFeatureProcessor(self.df)
        result = tcp.calculate_burstiness(features_list)
        self.assertEqual(expected_result, result)

    def test_max(self):
        # Test case where op = 'max'
        features_list = [1, 2, 3, 4, 5]
        expected_output = 0.6
        tcp = TcpIpFeatureProcessor(self.df)
        self.assertAlmostEqual(tcp.calculate_ratio(features_list, 'max'), expected_output, delta=0.01)

    def test_total(self):
        # Test case where op = 'total'
        features_list = [1, 2, 3, 4, 5]
        expected_output = 0.33333  # (max = 5, sum = 15, ratio = 5/15)
        tcp = TcpIpFeatureProcessor(self.df)
        self.assertAlmostEqual(tcp.calculate_ratio(features_list, 'total'), expected_output, delta=0.01)

    def test_zero(self):
        # Test case where all features are zero
        features_list = [0, 0, 0, 0]
        expected_output = 0
        tcp = TcpIpFeatureProcessor(self.df)
        self.assertEqual(tcp.calculate_ratio(features_list, 'max'), expected_output)
        self.assertEqual(tcp.calculate_ratio(features_list, 'total'), expected_output)

    def test_invalid_op(self):
        # Test case where op is invalid
        features_list = [1, 2, 3, 4, 5]
        tcp = TcpIpFeatureProcessor(self.df)
        with self.assertRaises(ValueError):
            tcp.calculate_ratio(features_list, 'invalid_op')

    def test_split_column_value_into_dummy_columns_with_description(self):
        tls = TLSFeatureProcessor(self.df)
        self.df = tls.split_column_value_into_dummy_columns('ext1', 'ext', ['23', '47802'])

        # Check that the new columns were created
        self.assertTrue('ext1_ext_23' in self.df.columns)
        self.assertTrue('ext1_ext_47802' in self.df.columns)
        # Test that new columns have the expected values
        self.assertListEqual(list(self.df['ext1_ext_23']), [1, 0])
        self.assertListEqual(list(self.df['ext1_ext_47802']), [1, 1])

    def test_split_column_value_into_dummy_columns(self):
        tls = TLSFeatureProcessor(self.df)

        # Test that the function correctly splits the column into dummy columns
        list_values = ['0', '23', '15', '65281']
        self.df = tls.split_column_value_into_dummy_columns('ext1', 'ext', list_values)

        # Check that the new columns were created
        self.assertTrue('ext1_ext_0' in self.df.columns)
        self.assertTrue('ext1_ext_23' in self.df.columns)
        self.assertTrue('ext1_ext_15' in self.df.columns)
        self.assertTrue('ext1_ext_65281' in self.df.columns)
        # Test that new columns have the expected values
        self.assertListEqual(list(self.df['ext1_ext_0']), [1, 1])
        self.assertListEqual(list(self.df['ext1_ext_23']), [1, 0])
        self.assertListEqual(list(self.df['ext1_ext_15']), [0, 1])
        self.assertListEqual(list(self.df['ext1_ext_65281']), [1, 0])


if __name__ == '__main__':
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestFunctions)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if result.wasSuccessful():
        sys.exit(0)
    else:
        sys.exit(1)
