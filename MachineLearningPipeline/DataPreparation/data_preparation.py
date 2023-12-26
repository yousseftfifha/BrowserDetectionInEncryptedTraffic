from DataPreparation.session_feature_extraction import PacketProcessor
from DataPreparation.tcp_ip_feature_engineering import TcpIpFeatureProcessor
from DataPreparation.tls_feature_engineering import TLSFeatureProcessor
from DataPreparation.data_cleaning import DataCleaning
import pandas as pd
import numpy as np
import time
import multiprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from utils import (load_data, save_pickle, write_list2file, encode_uan)
import warnings
warnings.filterwarnings('ignore')


class DataPreparation:
    def __init__(self, target, path, dest, test_size, val_size, cpu, dataframe=None):
        self.target = target
        self.path = path
        self.dest = dest
        self.test_size = test_size
        self.val_size = val_size
        self.cpu = cpu
        self.df = dataframe

    @staticmethod
    def preprocess_chunk(df_chunk, config):
        """
        The function ingest a chunk of the data to preprocess; (Session Processing & TCP/IP Processing)
        """
        packet_processor = PacketProcessor(df_chunk)
        df_chunk = packet_processor.select_generate_cols_by_packets(config)
        tcp_ip_processor = TcpIpFeatureProcessor(df_chunk)
        df_chunk = tcp_ip_processor.preprocess_tcpip_features(config)
        return df_chunk

    @staticmethod
    def clean_scale_export_data(cleaner, scaler, data, target_col, dest_path, set_name):
        """
        Given a fitted cleaner, the function perform cleaning functions as well as export the clean scaled data
        """
        x_cleaned, y_cleaned = cleaner.transform(data.drop([target_col], axis=1), data[target_col])
        x_cleaned[target_col] = y_cleaned
        # Export the cleaned data
        cleaned_file_path = f"{dest_path}/cleaned_{set_name}_set.csv"
        x_cleaned.to_csv(cleaned_file_path, sep=",", index=False)
        # Scale the numeric columns
        x_cleaned[cleaner.numeric_cols] = scaler.fit_transform(x_cleaned[cleaner.numeric_cols])
        # Export the scaled data
        scaled_file_path = f"{dest_path}/scaled_{set_name}_set.csv"
        x_cleaned.to_csv(scaled_file_path, sep=",", index=False)
        return x_cleaned

    def extract_features(self, data, config):
        """
        This function leverages multi threading usage to process session extraction and TCP/IP processing by splitting
        the data into as many chunks as available CPUs. TLS features are processed separately to extract unique values
        needed.
        """
        pool = multiprocessing.Pool(processes=int(self.cpu))
        print(f"----------- Processing Packets and Generating TCP/IP Features -----------")
        data = pd.concat([p.get() for p in [pool.apply_async(self.preprocess_chunk, args=(df_chunk, config))
                                            for df_chunk in np.array_split(data, int(self.cpu))]])
        tls_processor = TLSFeatureProcessor(data)
        data = tls_processor.process_tls_features(config)
        data = data.drop(columns=config['undesirable'], errors='ignore')  # Drop the obsolete features
        data = data[sorted(data.columns, key=lambda x: x != 'uan')]
        return data

    def preprocess_data(self, config):
        """
        Load the datasets, concat them and extract all necessary features.
        """
        start = time.time()
        self.df = load_data(self.path, config)
        self.df = encode_uan(self.df)
        self.df = self.extract_features(self.df, config)
        self.df.reset_index(drop=True, inplace=True)
        self.df.to_csv(self.dest + 'data.csv', sep=",")
        end = time.time()
        print(f"Total Elapsed Time for preprocess_data : {end - start} seconds")

    def split_clean_data(self):
        """
        Split the dataset into 3 subset; train (56 %), test (30 %), validation (14 %). The cleaning functions
        are applied to the train first then for the other sets
        """
        start = time.time()
        print('----------- Splitting Data -----------')
        # Splitting the data into 70% train and 30% test:
        data_train, data_test = train_test_split(self.df, test_size=float(self.test_size), shuffle=True,
                                                 random_state=np.random.RandomState(42), stratify=self.df[self.target])
        # Splitting the train into 80% train and 20% validation:
        data_train, data_val = train_test_split(data_train, test_size=float(self.val_size), shuffle=True,
                                                random_state=np.random.RandomState(42),
                                                stratify=data_train[self.target])
        data_train.reset_index(drop=True, inplace=True)
        data_val.reset_index(drop=True, inplace=True)
        data_test.reset_index(drop=True, inplace=True)

        # Cleaning and Scaling setup
        cleaner = DataCleaning(rare_thresh=0.01, quasi_constant_thresh=0.9, missing_thresh=0.2, encode_categ_cols=True)
        # Clean, scale, and export train data
        print('----------- Cleaning Train Data -----------')
        x_train, y_train = cleaner.fit_transform(data_train.drop([self.target, 'source'], axis=1),
                                                 data_train[self.target])
        save_pickle(cleaner, self.dest + 'cleaner.pkl')
        x_train[self.target] = y_train
        x_train.to_csv(self.dest + "cleaned_train_set.csv", sep=",")  # Export clean train

        new_features = x_train.columns.tolist()
        write_list2file(new_features, self.dest + 'feature_list.csv')  # Export the list of features

        scaler = StandardScaler()
        x_train[cleaner.numeric_cols] = scaler.fit_transform(x_train[cleaner.numeric_cols])  # Perform Scaling on the train
        save_pickle(scaler, self.dest + 'scaler.pkl')
        x_train.to_csv(self.dest + "scaled_train_set.csv", sep=",")  # Export the scaled train set

        print('----------- Cleaning Val Data -----------')
        x_val = self.clean_scale_export_data(cleaner, scaler, data_val, self.target, self.dest, 'validation')

        print('----------- Cleaning Test Data -----------')
        x_test = self.clean_scale_export_data(cleaner, scaler, data_test, self.target, self.dest, 'test')

        end = time.time()
        print(f"Total Elapsed Time for split_clean_data : {end - start} seconds")
        return x_train, x_val, x_test

    def prepare_clean_data(self, config):
        """
        Load, Extract, Preprocess and Clean the data. Returns 3 treated sets.
        """
        self.preprocess_data(config)
        return self.split_clean_data()
