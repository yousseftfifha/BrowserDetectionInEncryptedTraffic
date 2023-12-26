import time

from utils import *


class TLSFeatureProcessor:
    def __init__(self, df):
        self.df = df

    @staticmethod
    def check_cipher(json_data: str, cipher: str) -> dict:
        """
        Perform text preprocessing ti extract cipher from a nested malformed json file
        :param json_data: Data in the form of json
        :param cipher: Str
        :return: Dictionary
        """
        dict = {
            'tls.cipher': cipher,
            'new_tls.cipher_kex_algorithm': re.sub('"', '', json_data[5][19:]),
            'new_tls.cipher_auth_algorithm': re.sub('"', '', json_data[6][19:]),
            'new_tls.cipher_enc_algorithm': re.sub('"', '', json_data[7][19:]),
            'new_tls.cipher_hash_algorithm': re.sub('"', '', json_data[8][19:]),
            'new_tls.cipher_security': re.sub('"', '', json_data[9][14:])
        }
        return dict

    @staticmethod
    def extract_cipher(ciphers: list, config: dict) -> list[dict]:
        """
        Extract ciphers from Database if they exist else Scrape them from an official API
        :param ciphers: List of ciphers
        :param config: Config file
        :return: Dictionary
        """
        cipher_dict = []
        if not os.path.exists('config/ciphers_DB.csv'):
            with open('config/ciphers_DB.csv', 'a') as file:
                file.write(
                    'tls.cipher,new_tls.cipher_kex_algorithm,new_tls.cipher_auth_algorithm,'
                    'new_tls.cipher_enc_algorithm,new_tls.cipher_hash_algorithm,new_tls.cipher_security\n')
        for i in ciphers:
            cipherDB = pd.read_csv('config/ciphers_DB.csv')
            if i in cipherDB['tls.cipher'].unique():
                print('Getting Cipher from DB ', i)
                cipher_dict.append(cipherDB[cipherDB['tls.cipher'] == i].to_dict('records')[0])
            else:
                if requests.get(config['cipher_api_url'] + i).status_code != 404:
                    print('Scrapping Cipher ', i)
                    json = pd.read_csv(config['cipher_api_url'] + i)
                    cipher = TLSFeatureProcessor.check_cipher(json.columns, i)
                    cipher_dict.append(cipher)
                    with open('config/ciphers_DB.csv', 'a', newline='') as csvfile:
                        writer = csv.DictWriter(csvfile, fieldnames=cipher.keys())
                        writer.writerow(cipher)
                else:
                    print(i, ' Cipher NOT FOUND')
                    config['cipher_not_found'].update({'tls.cipher': i})
                    cipher_dict.append(config['cipher_not_found'])
        return cipher_dict

    @staticmethod
    def scrape_tls_dict(config: dict) -> any:
        """
        Extract a dictionary from a url
        :param config: Dictionary
        :return: Dictionary
        """
        if not os.path.exists('config/tls_dict.yaml'):
            print("----------- Scrapping TLS Extensions and Handshake types Dictionary from IANA-----------")
            for key, value in config.items():
                tls_dict = scrape_table(value['url'], value['id'])
                scraped_config = {key: tls_dict.set_index('Value').to_dict()[tls_dict.columns[1]]}
                with open('config/tls_dict.yaml', 'a') as file:
                    yaml.dump(scraped_config, file)
        return load_config('config/tls_dict.yaml')

    def process_extension_type(self, config: dict, ht: str, i: int) -> pd.DataFrame:
        """
        Process the handshake extension types for the server hello and client hello
        :param config: Dictionary
        :param ht: str Handshake type [client, server]
        :param i: int Handshake type [1: Client Hello, 2: Server Hello]
        :return: New Handshake type extension for the reduced handshake types
        """
        print('Processing ' + ht + ' Handshake')

        handshake_type = self.df.apply(
            lambda x: res(x['tls.handshake.type'], x['tls.handshake.extension.type'])[str(i)],
            axis=1)
        self.df["new_tls.handshake_type_" + ht] = handshake_type

        extension_type = pd.get_dummies(handshake_type.apply(pd.Series).stack()).sum(level=0)
        extension_type.columns = [
            'new_tls.handshake.extension.type_' + ht + '_' + get_description_extension(c, config) for c in
            extension_type.columns]
        handshake_type_data = pd.concat([self.df, extension_type], axis=1)

        return handshake_type_data.groupby(handshake_type_data.columns, axis=1).sum()

    def process_extension_length(self, ht: str, i: int) -> pd.DataFrame:
        """
        Process the extension length
        :param ht: str ['client','server']
        :param i: Handshake type [1: Client Hello, 2: Server Hello]
        :return:
        """
        print('Processing ' + ht + ' Handshake Extension Length')

        self.df = str_to_list(self.df, ['tls.handshake.extensions_length'])

        handshake_length = self.df.apply(
            lambda x: res(x['tls.handshake.type'], x['tls.handshake.extensions_length'])[str(i)], axis=1)

        self.df["new_tls.handshake_length_" + ht] = handshake_length
        self.df["new_tls.handshake_length_" + ht] = pd.to_numeric(self.df["new_tls.handshake_length_" + ht],
                                                                  errors='coerce')

        return self.df

    def process_handshake_type_ext(self, config: dict) -> pd.DataFrame:
        """
        Process the handshake type extension and lengths and generate the new features
        :param config: Dictionary
        :return: Dataframe
        """
        start = time.time()
        self.df["tls.handshake.extension.type"] = self.df["tls.handshake.extension.type"].map(
            convert_str_list_of_list)

        self.df['tls.handshake.type'] = self.df['tls.handshake.type'].map(convert_str_list_to_list)
        self.df['tls.handshake.type'] = self.df['tls.handshake.type'].map(list_of_list_to_list)
        for i in zip(['client', 'server'], [1, 2]):
            self.df = self.process_extension_type(config, i[0], i[1])
            self.df = self.process_extension_length(i[0], i[1])

        end = time.time()
        print('Elapsed Time:', end - start, 'seconds')
        return self.df

    def decompose_ciphers(self, config: dict) -> pd.DataFrame:
        """
        Extract High level of information from the ciphers
        :param config: Config file
        :return: Dataframe
        """
        cipher_decomposition_dict = self.extract_cipher(self.df['tls.cipher'].unique(), config)
        cipher_decomposition = pd.DataFrame(cipher_decomposition_dict)
        self.df = pd.merge(self.df, cipher_decomposition, on='tls.cipher', how='inner')
        return self.df

    def process_cipher(self, cipherlist: list, GreaseList: list, config: dict) -> pd.DataFrame:
        """
        Perform all operations in relation with the cipher
        :param GreaseList: List containing grease-like ciphers
        :param config: Config file
        :return: Dataframe
        """
        self.df = self.preprocess_list_ignore_order('tls.handshake.ciphersuite.ch', cipherlist, 'cipher')
        self.df = sum_columns_with_suffix(self.df, GreaseList, 'new_tls.handshake.ciphersuite.ch_presence_cipher_')
        self.df = self.decompose_ciphers(config)
        return self.df

    def preprocess_version(self, column_name: str, supported_version_col: str, ch_version_col: str) -> pd.DataFrame:
        """
        Generate a new feature column 'new_column_name' in the given dataframe
        based on the supported version and client hello version features
        :param column_name: string, name of the new feature column to be created
        :param supported_version_col: string, name of the column containing supported version
        :param ch_version_col: string, name of the column containing client hello version
        :return: DataFrame
        """

        new_supported_version_col = 'new_' + supported_version_col
        self.df[new_supported_version_col] = self.df[supported_version_col].fillna(0)
        tls_version = []
        for supported_versions, ch_version in zip(self.df[new_supported_version_col], self.df[ch_version_col]):
            if supported_versions != 0:
                supported_versions = ast.literal_eval(supported_versions)
                # Keep only versions containing 'TLS'
                supported_versions = [v for v in supported_versions if 'TLS' in v]
                if supported_versions:
                    # Get the highest TLS version number
                    highest_version = sorted(supported_versions, reverse=True)[0]
                    tls_version.append(highest_version)
                else:
                    # If no TLS version in the supported version list, use the client hello version
                    tls_version.append(ch_version)
            else:
                # If the supported version column is empty, use the client hello version
                tls_version.append(ch_version)
        # Add the new feature column to the dataframe
        self.df[column_name] = tls_version
        return self.df

    def split_column_value_into_dummy_columns(self, init_column: str, suffix: str, list_values: list) -> pd.DataFrame:
        """
        This function splits the values (initially list) of a column between set of dummy columns
        :param init_column: initial column name
        :param suffix: string suffix used to name the new columns
        :param list_values: list of unique values to create a dummy column for each
        :return: DataFrame
        """
        # Create a dictionary that maps each list value to a new column name
        column_map = {val: f"{init_column}_{suffix}_{val}" for val in list_values}
        # Create a dataframe with binary columns for each list value
        dummy = pd.DataFrame(self.df[init_column].apply(lambda x: [val in x for val in list_values]).tolist(),
                             columns=column_map.values(), index=self.df.index)
        dummy = dummy.astype(bool)
        # Merge the dummy dataframe back into the original dataframe
        self.df = pd.concat([self.df, dummy], axis=1)

        return self.df

    def preprocess_supported_group(self, supported_group_list: list, greaselist: list) -> pd.DataFrame:
        """
        Returns the number of supported groups per session and creates dummies
        :param supported_group_list: list of possible supported groups
        :param greaselist: list of grease values
        :return: Dataframe
        """
        start = time.time()
        self.df = self.preprocess_list_ignore_order('tls.handshake.extensions_supported_group.ch', supported_group_list,
                                                    'group')
        self.df = sum_columns_with_suffix(self.df, greaselist, 'new_tls.handshake.extensions_supported_group'
                                                               '.ch_presence_group_')
        end = time.time()
        print('Elapsed Time:', end - start, 'seconds')
        return self.df

    def preprocess_sig_hash_alg(self, supported_sighash_list: list, greaselist: list) -> pd.DataFrame:
        """
        Returns the number of supported groups per session and creates dummies
        :param supported_sighash_list: list of possible supported groups
        :param greaselist: list of grease values
        :return: Dataframe
        """
        start = time.time()
        self.df = self.preprocess_list_ignore_order('tls.handshake.sig_hash_alg.ch', supported_sighash_list, 'sha')
        self.df = sum_columns_with_suffix(self.df, greaselist, 'new_tls.handshake.sig_hash_alg.ch_presence_sha_')
        end = time.time()
        print('Elapsed Time:', end - start, 'seconds')
        return self.df

    def preprocess_list_track_order(self, column_name: str, nb_col: int, capture_last = True) -> pd.DataFrame:
        """
        Process the list of ciphersuites
        :param column_name: Name of the column
        :param nb_col: Number of dummies to make
        :param capture_last: bool to capture the last element of the values
        :return: Exploding dummies from the list of ciphersuites and their number
        """
        new_col = 'new_' + column_name + '_index'
        self.df[new_col] = self.df[column_name].map(convert_str_list_to_list)
        for i in range(1, nb_col + 1):
            self.df[new_col + '_' + str(i)] = self.df[new_col].map(lambda x: x[i] if len(x) > i else 'undefined')
        self.df[new_col + '_nb'] = list(self.df[new_col].map(lambda x: len(x)))
        if capture_last:
            self.df[new_col + '_last'] = self.df[new_col].map(lambda x: x[-1] if len(x) > 0 else 'undefined')
            self.df[new_col + '_secondlast'] = self.df[new_col].map(lambda x: x[-2] if len(x) > 1 else 'undefined')
            self.df[new_col + '_thirdlast'] = self.df[new_col].map(lambda x: x[-3] if len(x) > 2 else 'undefined')

        return self.df

    def preprocess_list_ignore_order(self, column_name: str, config_values: list, prefix: str) -> pd.DataFrame:
        """
        Create dummy columns based on the presence of unique categories
        :param column_name: name of the column to process
        :param config_values: list of values that we will create a dummy column
        :param prefix: prefix used to generate the new columns
        :return:
        """
        new_col = 'new_' + column_name
        print(column_name, " ", config_values, len(config_values))
        self.df[new_col] = self.df[column_name].map(convert_str_list_to_list)
        self.df[new_col + '_nb'] = list(self.df[new_col].map(lambda x: len(x)))
        self.df = self.split_column_value_into_dummy_columns(new_col, 'presence_' + prefix, config_values)
        return self.df

    def process_tls_features(self, config: dict) -> pd.DataFrame:
        """
        Process all TLS features
        :param config: Config file
        :return: Dataframe
        """
        print('----------- Processing TLS features -----------')
        tls_dict = self.scrape_tls_dict(config['tls_scrapping'])

        print('----------- Processing Handshake Types and extension -----------')
        self.df = self.process_handshake_type_ext(tls_dict['extension_types'])

        print('----------- Processing Cipher Suites -----------')
        cipher_suites_vocab = get_unique_categories(self.df, 'tls.handshake.ciphersuite.ch')
        self.df = self.process_cipher(cipher_suites_vocab, config['grease_dict'], config)

        print('----------- Processing Supported Groups -----------')
        supp_group_vocab = get_unique_categories(self.df, 'tls.handshake.extensions_supported_group.ch')
        self.df = self.preprocess_supported_group(supp_group_vocab,config['grease_dict'])

        print('----------- Processing Sig Hash Algorithm -----------')
        sig_hash_alg_vocab = get_unique_categories(self.df, 'tls.handshake.sig_hash_alg.ch')
        self.df = self.preprocess_sig_hash_alg(sig_hash_alg_vocab,config['grease_dict'])

        print('----------- Processing ALPN Extension-----------')
        self.df = self.preprocess_list_track_order('tls.handshake.extensions_alpn_str.ch', 2, False)

        print('----------- Processing Supported Version -----------')
        self.preprocess_version('tls.version', 'tls.handshake.extensions.supported_version.ch',
                                'tls.handshake.version.ch')
        return self.df
