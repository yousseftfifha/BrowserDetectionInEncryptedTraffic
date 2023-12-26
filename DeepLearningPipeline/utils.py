import yaml
import ast
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")


def load_config(config_filename):
    """
    Load the confing file
    :param config_filename: Str indicating the name of the config file
    :return: Configuration file
    """
    with open(config_filename) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def convert_str_list_to_list(value):
    """
    Convert a string list to list of list: "[str,str,str]"  to [str,str,str]
    :param value: Str indicating the value to change
    :return: List
    """
    liste = ast.literal_eval(value)
    if liste == ['']:
        return []
    else:
        return liste


def str_to_list(df, cols):
    """
    Convert a string list to a list: "[elem1,elem2,elem3,elem4]" to [elem1,elem2,elem3,elem4]
    :param df: Dataframe
    :param cols: List of columns containing str list to list
    :return: Dataframe
    """
    for col in cols:
        df[col] = df[col].apply(lambda x: ast.literal_eval(x) if type(x) == str else x)
    return df


def list_of_list_to_list(list_of_lists):
    """
    Convert a list of lists to a list: "[[elem],[elem],[elem]]" to [[elem],[elem],[elem]]
    :param list_of_lists: List of list
    :return: List
    """
    liste = []
    for sublist in list_of_lists:
        for item in sublist:
            liste.append(item)
    return liste


def convert_str_list(value, to_integer=False):
    """
    This function convert a string value in the format : '['int','int', ...'int']'  to a list of int
    :param value:
    :param to_integer:
    :return:
    """
    try:
        converted_list = ast.literal_eval(value)
        if converted_list == ['']:
            return []
        else:
            if to_integer:
                converted_list = [eval(i) if i != '' else None for i in converted_list]
            return converted_list
    except Exception as e:
        print("Error ", e, value)


def convert_hexa_str_list_to_list_int(value):
    """
    Convert a string list of hexadecimal values to a list of int: "[0x,0x]" to [0,0]
    :param value: HexaDecimal value
    :return: List
    """
    liste = ast.literal_eval(value)
    if liste == ['']:
        return []
    else:
        liste = [str(int(i, 16)) if i != '' else None for i in liste]
        return liste


def convert_str_list_of_list(value):
    """
    Convert a string list of list to a list of list of int: "[['int'],['int']]" to [int,int]
    :param value: str list of lists
    :return: List
    """
    list_of_list = ast.literal_eval(value)
    new_list = [[] if len(l) == 1 and l[0] == '' else [eval(j) if j != '' else None for j in l] for l in list_of_list]
    return new_list


def find_ht(row, ht_column, ht_value):
    """
    This function return the index of a given handshake type
    :param row: the row to apply on
    :param ht_column: handshake type column name
    :param ht_value: handshake type value (1 : hello client, 2: server hello)
    :return: the index of the given handshake type value
    """
    try:
        return next(idx for idx, elem in enumerate(row[ht_column]) if ht_value in elem)
    except:
        return -1


def get_hc_index(data, ht_column):
    """
    This function create a column indicating the index of packet containing HC message
    :param ht_column: handshake type column
    :param data: dataframe
    :return: new_col
    """
    new_col = data.apply(find_ht, args=(ht_column, '1'), axis=1)

    return new_col


def select_packets_by_hello_client(data, column, hc_pos_column, nb_max_packets):
    """
    This function return new filtered column starting with hello client and limited by the nb_max_packets
    :param data: dataframe
    :param column: column to filter
    :param hc_pos_column: hello client position
    :param nb_max_packets: maximum number of packets
    :return: new filtered column
    """

    new_col = data.apply(lambda x: x[column][x[hc_pos_column]: x[hc_pos_column] + nb_max_packets], axis=1)

    return new_col


def res(keys, values):
    """
    takes a combination of a key/value and return a dictionary
    :param keys: key
    :param values: value
    :return: Dictionary
    """
    return {k: v for k, v in zip(keys, values)}
