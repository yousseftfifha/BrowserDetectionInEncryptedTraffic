import pandas as pd
import ast
import yaml
import pickle
import csv
import requests
import re
import os
from bs4 import BeautifulSoup
from pathlib import Path


def load_data(path: str, config: dict) -> pd.DataFrame:
    """
    Load data given a path and subtract only features given in config
    :param path: Str containing the path for the folder containing the data
    :param config: dictionary
    """
    datasets = []  # List to assemble the processed dataframes
    for p in Path(path).glob('*.csv'):
        print(f"Loading {p.name} ...")
        data = pd.read_csv(path + p.name, sep=',', low_memory=False, usecols=config['list_of_features'])
        data['source'] = p.stem
        data = data.sample(100)
        data.reset_index(drop=True, inplace=True)
        datasets.append(data)
    return pd.concat(datasets, join='outer', ignore_index=True)


def find_index(row: str, col: str, value: int) -> int:
    """
    Finds the index of a value given in parameter inside a list of values
    :param row: Str containing the row to apply the function on
    :param col: Str containing the column name to apply the function on
    :param value: handshake type value (1: Hello Client, 2: Hello Server )
    :return: the index of the given handshake type value
    """
    try:
        return next(idx for idx, elem in enumerate(row[col]) if value in elem)
    except:
        return -1


def get_handshake_type_index(data: pd.DataFrame, handshake_type_list: list, ht_column: str):
    """
    Creates a column indicating the index of the handshake type
    :param data: Dataframe
    :param handshake_type_list: List
    :param ht_column: Str indicating the handshake type column
    :return:
    """
    for ht in handshake_type_list:
        data['ht_pos' + str(ht)] = data.apply(find_index, args=(ht_column, ht), axis=1)


def convert_str_list_of_list(value: str) -> list:
    """
    Convert a string list_of_list to a list_of_list: "[['int'],['int']]" to [['int'],['int']]
    :param value: str list of lists
    :return: List
    """
    list_of_list = ast.literal_eval(value)
    new_list = [[] if len(l) == 1 and l[0] == '' else [eval(j) if j != '' else None for j in l] for l in list_of_list]
    return new_list


def convert_str_list_to_list_int(value: str) -> list:
    """
    This function convert a string value in the format : '['int','int', ...'int']'  to a list of int
    :param value: str list of lists
    :return: List
    """
    liste = ast.literal_eval(value)
    if liste == ['']:
        return []
    else:
        liste = [eval(i) if i != '' else None for i in liste]
        return liste


def convert_str_list_to_list(value: str) -> list:
    """
    This function converts a string value in the format : '[str,sr, ...str]'  to a list of str
    :param value: str list
    :return: List    """
    if value:
        liste = ast.literal_eval(value)
        if liste == ['']:
            return []
        else:
            return liste
    else:
        print("nan value:", value)
        return []


def str_to_list(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    Convert a string list to a list: "[elem1,elem2,elem3,elem4]" to [elem1,elem2,elem3,elem4]
    :param df: Dataframe
    :param cols: List of columns containing str list to list
    :return: Dataframe
    """
    for col in cols:
        df[col] = df[col].apply(lambda x: ast.literal_eval(x) if type(x) == str else x)
    return df


def list_of_list_to_list(list_of_lists: list) -> list:
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


def convert_hexa_str_list_to_list_int(value: str) -> list:
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


def load_config(config_filename: str) -> any:
    """
    Load the confing file
    :param config_filename: Str indicating the name of the config file
    :return: Configuration file
    """
    with open(config_filename) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def get_unique_categories(data: pd.DataFrame, column_name: str) -> list:
    """
    Search in a given dataframe the unique categories for a certain column
    :param data: input Dataframe
    :param column_name: Str name of the column
    :return: list, unique categories
    """
    unique_categories = set()
    try:
        data['temp'] = data[column_name].map(convert_str_list_to_list)
    except:
        pass
    for row in data['temp']:
        unique_categories.update(row)
    return list(unique_categories)


def res(keys: str, values: str) -> dict:
    """
    takes a combination of a key/value and return a dictionary
    :param keys: key
    :param values: value
    :return: Dictionary
    """
    return {k: v for k, v in zip(keys, values)}


def get_description_extension(column_name: str, extension_type_dict: dict) -> str:
    """
    Map the value of extension with its relative description
    :param column_name: Str indicating the column
    :param extension_type_dict: Dictionary containing the description of each extension type
    :return: Str name of the extension mapped
    """
    try:
        if column_name in extension_type_dict.keys():
            return extension_type_dict[column_name]
        else:
            for k in extension_type_dict.keys():
                value = [int(s) for s in k.split('-')]
                if int(column_name) >= value[0] and int(column_name) <= value[-1]:
                    return extension_type_dict[k]
            return 'No_Extension_mapped'
    except:
        return 'No_value'


def save_pickle(object_to_save: any, file_path: str):
    """
    This function saves a pickle file of the specified object at the specified file path.
    :param object_to_save: The object to be saved as a pickle file.
    :param file_path: The file path where the pickle file will be saved.
    :return:
    """
    with open(file_path, 'wb') as f:
        pickle.dump(object_to_save, f)


def load_pickle(file_path: str) -> any:
    """
     This function loads a pickle file from the specified file path and returns the object it contains.
    :param file_path: the file path of the pickle file to be loaded.
    :return: The object contained in the pickle file.
    """
    with open(file_path, 'rb') as f:
        object_to_read = pickle.load(f)
        return object_to_read


def write_list2file(list2write: list, file_path: str):
    """
    Write a file from a list
    :param list2write: the list of export
    :param file_path: the destination path
    :return:
    """
    csv_file = open(file_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    for l in list2write:
        csv_writer.writerow([l])
    csv_file.close()


def sum_columns_with_suffix(df: pd.DataFrame, column_list: list, col: str) -> pd.DataFrame:
    """
    Sums the columns in a DataFrame that have specified suffixes and stores the result in a new column.
    :param df: input DataFrame
    :param column_list: A list of column names or suffixes to match
    :param col: The name of the new column to store the sum.

    :return: The modified DataFrame with the matching columns summed and the original columns dropped.
    """
    pattern = r'({})$'.format('|'.join(column_list))
    matching_columns = [col for col in df.columns if re.search(pattern, col)]
    df[col + 'grease'] = df[matching_columns].sum(axis=1)
    df.drop(columns=matching_columns, inplace=True)
    return df


def scrape_table(url: str, table_id: str) -> pd.DataFrame:
    """
    Scrape a website using beautiful soup
    :param url: desired url to scrape
    :param table_id: id of the table
    :return: Dictionary containing the scraped data
    """
    session = requests.Session()
    response = session.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    table = soup.find('table', id=table_id)
    headers = [th.text.strip() for th in table.find_all('th')]
    rows = [[td.text.strip() for td in tr.find_all('td')] for tr in table.find_all('tr') if tr.find_all('td')]
    df = pd.DataFrame(rows, columns=headers)
    df.iloc[:, 1] = df.iloc[:, 1].str.replace(r'\(.*?\)', '').str.strip()
    df.loc[df.iloc[:, 1] == 'Reserved', df.columns[1]] = 'grease'

    return df

def config_log(log_dir: str):
    """
    Create a directory if it doesn't exist
    :param log_dir: str, name of the directory
    :return:
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)


def encode_uan(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode the target label
    :param df: input Dataframe
    :return: Dataframe
    """
    uan_mapping = {'firefox': 0, 'chrome': 1, 'opera': 2, 'edge': 3, 'internet-explorer': 4}
    df['uan'] = [uan_mapping.get(uan, 5) for uan in df['uan']]
    return df


def getXy(df: pd.DataFrame, label: str) -> tuple:
    """
    Split data into target and features
    :param df: Dataframe
    :param label: target label
    :return: tuple X & y
    """
    X = df.drop([label], axis=1)
    y = df.uan
    return X, y
