from utils import *


def get_data(directory, config):
    """
    Read data
    :param directory: path indicating the directory of the data
    :return: list of datasets with a list of their respective names
    """
    print('----------- Loading Raw Datasets -----------')
    datasets = []
    for p in Path(directory).glob('*.csv'):
        print(f"Loading {p.name} ....\n")
        df = pd.read_csv(directory + p.name, low_memory=False, sep=',', usecols=config)
        datasets.append(df)
    return pd.concat(datasets)


def format_features(data):
    data = str_to_list(data, ["tls.handshake.extension.type", "tls.handshake.extension.len"])
    data['tls.handshake.type'] = data['tls.handshake.type'].map(convert_str_list_to_list)
    data['tls.handshake.type'] = data['tls.handshake.type'].map(list_of_list_to_list)
    data['tls.handshake.extensions_supported_group.ch'] = data['tls.handshake.extensions_supported_group.ch'].map(
        convert_hexa_str_list_to_list_int)
    data['packet_directions'] = [convert_str_list(x, False) for x in data['packet_directions']]
    for feature in ['ip.ttl', 'ip.len', 'tcp.window_size_value', 'tcp.offset', 'frame.time_epoch']:
        print('----feature', feature)
        data[feature] = [convert_str_list(x, True) for x in data[feature]]
    data['tcp.flags'] = data['tcp.flags'].map(ast.literal_eval)
    return data


def sample_from_categories(df, category_col, n):
    categories = df[category_col].unique()
    sampled_data = []

    # Sample "n" instances from each category
    for category in categories:
        category_data = df[df[category_col] == category]
        samples = category_data.sample(n=min(n, len(category_data)), replace=False)
        sampled_data.append(samples)

    sampled_df = pd.concat(sampled_data)
    sampled_df.reset_index(drop=True, inplace=True)

    return sampled_df


def get_description_from_dict(extension_id, description_dict):
    """
    Gets the description of a column name from a dictionary of descriptions.

    :param extension_id: The id of the column whose description is to be retrieved.
    :param description_dict: A dictionary containing column names as keys and their descriptions as values.

    :return: (str) The description of the column name if found, otherwise 'Unknown'

    """
    try:
        if extension_id in description_dict.keys():
            return description_dict[extension_id]
        else:
            id_val = int(extension_id)
            for k in description_dict.keys():
                value = [int(s) for s in k.split('-')]
                if value[0] <= id_val <= value[-1]:
                    return description_dict[k]
            return 'Undefined'
    except:
        return 'Undefined'


def process_extension_type(df, ht, i):
    """
    Process the handshake extension types for the server hello and client hello
    :param df: Dataframe
    :param extension_type_dict: Dictionary
    :param ht:
    :param i: Handshake type [1: Client Hello, 2: Server Hello]
    :return: New Handshake type extension for the reduced handshake types
    """
    handshake_type = df.apply(lambda x: res(x['tls.handshake.type'], x['tls.handshake.extension.type'])[str(i)],
                              axis=1)
    df["new_tls.handshake_type_" + ht] = handshake_type
    return df


def process_extension_length(df, ht, i):
    """
    Process the extension length
    :param df: Dataframe
    :param ht: str ['client','server']
    :param i: Handshake type [1: Client Hello, 2: Server Hello]
    :return:
    """
    df = str_to_list(df, ["tls.handshake.extensions_length"])
    handshake_length = df.apply(
        lambda x: res(x['tls.handshake.type'], x['tls.handshake.extensions_length'])[str(i)], axis=1)

    df["final_new_tls.handshake_length_" + ht] = handshake_length
    # df["final_new_tls.handshake_length_" + ht] = pd.to_numeric(df["final_new_tls.handshake_length_" + ht],
    #                                                            errors='coerce')

    return df


def process_extension_type_len_type(df, ht, i):
    """
    Process the handshake extension types for the server hello and client hello
    :param df: Dataframe
    :param ht:
    :param i: Handshake type [1: Client Hello, 2: Server Hello]
    :return: New Handshake type extension for the reduced handshake types
    """
    try:
        handshake_type = df.apply(lambda x: res(x['tls.handshake.type'], x['tls.handshake.extension.len'])[str(i)],
                                  axis=1)
        df['tls.handshake.extension.len_' + ht] = handshake_type
        df.reset_index(drop=True, inplace=True)
    except:
        print('continue')
    return df


def convert_to_binary(df, col):
    print('-----', col)
    try:
        df[col] = df[col].map(convert_str_list_to_list)
    except:
        pass
    if col in ['final_new_tls.handshake_length_client', 'final_new_tls.handshake_length_server']:
        df[col] = df[col].astype(str).map(lambda x: bin(int(str(x), 16))[2:] if x != '' else bin(int(str(0), 16))[2:])
    else:
        binary = [
            ''.join(bin(int(str(elem), 16))[2:] if elem else bin(int(str(0), 16))[2:] for elem in row)
            for row in df[col]
        ]
        df[col] = binary

    return df


def create_matrices(data, cols, pad):
    all_matrices = []
    scaler = MinMaxScaler(feature_range=(-1, 1))

    for row in data[cols].itertuples(index=False):
        iat = np.cumsum(row[0])
        lengths = row[1]
        ws = row[2]
        direction = [0 if elem == 'O' else 1 for elem in row[3]]

        iat = np.pad(iat, (0, pad - len(iat)))
        lengths = np.pad(lengths, (0, pad - len(lengths)))
        ws = np.pad(ws, (0, pad - len(ws)))
        direction = np.pad(direction, (0, pad - len(direction)))

        lengths *= (-1) ** direction
        ws *= (-1) ** direction

        iat = scaler.fit_transform(iat.reshape(-1, 1)).flatten()
        lengths = scaler.fit_transform(lengths.reshape(-1, 1)).flatten()
        ws = scaler.fit_transform(ws.reshape(-1, 1)).flatten()

        matrix = np.column_stack((iat, lengths, ws)).reshape(-1, len(cols) - 1)
        all_matrices.append(matrix)
    data['timeseries'] = all_matrices

    return data


def process_handshake_type_ext(df, config):
    """
    Process the handshake type extension and lengths and generate the new features
    :param df: Dataframe
    :param config: Dictionary
    :return: Dataframe
    """
    for i in zip(['client', 'server'], [1, 2]):
        df = process_extension_type(df, i[0], i[1])
        df = process_extension_length(df, i[0], i[1])
        df = process_extension_type_len_type(df, i[0], i[1])
    print(df['final_new_tls.handshake_length_client'])
    binary_columns = ['final_new_tls.handshake_length_client', 'final_new_tls.handshake_length_server']
    df[binary_columns] = df[binary_columns].fillna(0)
    for column in config['convert_binary']:
        df = convert_to_binary(df, column)
    df['all'] = df[config['convert_binary']].astype(str).apply(''.join, axis=1)
    if all(len(row) == len(df['all'][0]) for row in df['all']):
        print("All rows in the column have the same length.")
    else:
        print("Rows in the column have varying lengths.")
        df['all'] = df['all'].str.pad(width=df['all'].str.len().max(), side='left', fillchar='0')


    return df


def filter_data(data, features, nb_packets):
    data['new_hc_index'] = get_hc_index(data, 'tls.handshake.type')  # Get the index of hello client

    for feature in features:
        data[feature] = select_packets_by_hello_client(data, feature, 'new_hc_index', nb_packets)
    for col in ['frame.time_epoch', 'ip.len', 'tcp.window_size_value', 'ip.ttl', 'tcp.offset', 'tcp.flags']:
        for direc in ['I', 'O']:
            filter_packet_in_direc(data, col, 'packet_directions', direc)
    return data

def calculate_diffs(features_list):
    if len(features_list) > 1:
        diff = np.insert(np.diff(features_list), 0, 0).tolist()
    else:
        diff = [0]
    return diff


def split_column_value_into_dummy_columns(df, init_column, suffix, list_values, description_dict=None):
    """
    This function splits the values (initially list) of a column between set of dummy columns

    Params:
        -df: dataframe
        -init_column: initial column name
        -suffix: string suffix used to name the new columns
        -list_values: list of unique values to create a dummy column for each

    """

    # Create a dictionary that maps each list value to a new column name
    column_map = {val: f"final_{init_column}_{suffix}_{val}" for val in list_values}
    # Create a dataframe with binary columns for each list value
    dummy = pd.DataFrame(df[init_column].apply(lambda x: [val in x for val in list_values]).tolist(),
                         columns=column_map.values(), index=df.index)
    dummy = dummy.astype(bool)
    # Merge the dummy dataframe back into the original dataframe
    df = pd.concat([df, dummy], axis=1)
    if description_dict is not None:
        # create a dictionary to map the old column names to new ones
        column_mapping = {
            old_col: f"final_{init_column}_{get_description_from_dict(old_col.split('_')[-1], description_dict)}"
            for old_col in dummy.columns}
        # rename the columns using the mapping dictionary
        df = df.rename(columns=column_mapping)

    return df


def filter_packet_in_direc(df, column_name, direction_column, direction='O'):
    """
    Extract Forward and Backward features
    :param df: Dataframe
    :param column_name: Str
    :param direction_column: Str
    :param direction: [I:in Backward, O:out Forward]
    :return: Dataframe
    """
    if direction not in ['O', 'I']:
        print("Wrong direction")
    else:
        feature_direc = []
        for i in range(len(df)):
            if direction in df.loc[i, direction_column]:
                feature_direc.append(
                    [x for x, y in zip(df.loc[i, column_name], df.loc[i, direction_column]) if y == direction])
            else:
                feature_direc.append([])
        df[column_name + '_' + direction] = feature_direc


def process_tcp_ip_features(data):
    data['frame.time_epoch'] = data['frame.time_epoch'].apply(calculate_diffs)
    data['frame.time_epoch_O'] = data['frame.time_epoch_O'].apply(calculate_diffs)
    data['frame.time_epoch_I'] = data['frame.time_epoch_I'].apply(calculate_diffs)

    data['tcp.flags_count_I'] = [[len(l) for l in col_val] for col_val in data['tcp.flags_I']]
    data['tcp.flags_count_O'] = [[len(l) for l in col_val] for col_val in data['tcp.flags_O']]

    for list_col in ['frame.time_epoch', 'ip.len', 'tcp.window_size_value', 'ip.ttl', 'tcp.offset', 'tcp.flags_count']:
        for direc in ['I', 'O']:
            data['temp_col'] = data[list_col + '_' + direc].map(lambda x: list(filter(lambda val: val is not None, x)))
            new_col = 'stat_' + list_col + '_' + direc

            commun_op = ['mean', 'min', 'max', 'std', 'median', 'var']
            for op in commun_op:
                data[new_col + '_' + op] = [getattr(np, op)(x) if len(x) > 0 else None for x in data['temp_col']]
    for flag in ['PSH', 'RST', 'SYN', 'CWR', 'FIN', 'ECE', 'ACK']:
        data['stat_tcp.flags_' + direc + '_' + flag] = [sum([flag in list_f for list_f in val]) for val in
                                                        data['tcp.flags']]
    columns = ['frame.time_epoch', 'ip.len', 'tcp.window_size_value', 'packet_directions']

    data = create_matrices(data, columns, 20)
    return data
