import pandas as pd
import numpy as np
import argparse
import time
from utils import (load_config, config_log, getXy, write_list2file)
import logging
from FeatureSelection.voting_selection import VotingSelector
from ModelSelection.model_selection import ModelSelection
from DataPreparation.data_preparation import DataPreparation

parse = argparse.ArgumentParser(description="Passive Browser Detection in TLS Encrypted Traffic")

parse.add_argument("-target", dest="target", default='uan', help="target name")
parse.add_argument("-path", dest="path", default='C:\\data\\synth_browsers\\', help="data directory path")
parse.add_argument("-dest", dest="dest", default='C:\\data\\synth_browsers\\test\\', help="output directory path")
parse.add_argument("-testsize", dest="test_size", default='0.3', help="output directory path")
parse.add_argument("-valsize", dest="val_size", default='0.2', help="output directory path")
parse.add_argument("-cpu", dest="cpu", default='10', help="output directory path")
parse.add_argument("-log", dest="log", default='log\\', help="Log file path")

args, argList = parse.parse_known_args()

features_evals = []


def main():
    start = time.time()

    config = load_config('config/config.yaml')
    fs_config = config['feature_selection']['models']
    ms_config = config['model_selection']
    config_log(args.log)

    logging.basicConfig(filename=args.log + "browser_detection.log",
                        level=logging.INFO,
                        format='%(levelname)s:%(asctime)s:%(message)s')

    data_prep = DataPreparation(args.target, args.path, args.dest, args.test_size, args.val_size, args.cpu)
    train, val, test = data_prep.prepare_clean_data(config)

    print(train.uan.value_counts())

    x_train, y_train = getXy(train, args.target)
    x_val, y_val = getXy(val, args.target)
    x_test, y_test = getXy(test, args.target)

    voting_selector = VotingSelector(fs_config, x_train.select_dtypes(exclude=[np.number, bool]).columns.tolist(),
                                     x_train.select_dtypes(include=[np.number, bool]).columns.tolist())

    best_threshold = voting_selector.get_best_threshold(args.log, x_train, y_train, x_val, y_val)
    print(f"Final selected features: {best_threshold['features']}")

    write_list2file(best_threshold['features'], args.log + '/best_features_to_keep.csv')
    x_train, x_val, x_test = x_train[best_threshold['features'].values[0]], x_val[best_threshold[
        'features'].values[0]], x_test[best_threshold['features'].values[0]]

    model_selector = ModelSelection(ms_config['hyperparam_configs'], 'f1_micro', 'f1_score',
                                    ms_config['rgs_nb_iterations'],
                                    ms_config['Kfold'])

    best_model = model_selector.select_best_model(x_train, y_train, x_val, y_val, True)

    print(f"best model: {str(best_model)}")
    model_selector.model_results.to_csv(args.log + "/model_results.csv", sep=";")
    model_selector.evaluate(best_model, pd.concat([x_train, x_val]), x_test,
                            pd.concat([y_train, y_val]), y_test)

    end = time.time()
    print(f"Total Elapsed Time for browser detection : {end - start} seconds")


if __name__ == '__main__':
    main()
