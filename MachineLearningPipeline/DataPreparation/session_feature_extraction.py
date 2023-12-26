from utils import (convert_str_list_of_list, convert_str_list_to_list, convert_str_list_to_list_int,
                   get_handshake_type_index)


class PacketProcessor:
    def __init__(self, df):
        self.df = df

    def select_packets_by_nb_tcp_syn(self, column: str, hc_pos_column: str, nb_max_packets: int,
                                     start_from_tcp_syn: bool) -> str:
        """
        Return the new filtered column starting with hello client and limited by a certain number of packets
        :param column: Column to filter
        :param hc_pos_column: Hello client position
        :param nb_max_packets: maximum number of packets
        :param start_from_tcp_syn: Boolean indicating whether to take syn packets into consideration
        :return: the new filtered packet
        """
        if 'new' in column:
            new_col = column + '_' + str(nb_max_packets) + '_' + str(start_from_tcp_syn)
        else:
            new_col = 'new_' + column + '_' + str(nb_max_packets) + '_' + str(start_from_tcp_syn)

        if start_from_tcp_syn:
            self.df[new_col] = self.df.apply(lambda x: x[column][:nb_max_packets], axis=1)
        else:
            self.df[new_col] = self.df.apply(lambda x: x[column][x[hc_pos_column]: x[hc_pos_column] + nb_max_packets],
                                             axis=1)

        return new_col

    def get_packet_val(self, column: str, pos_column: str, new_col: str):
        """
        Extract from a list of values, the element at a position given in parameters
        :param column: Str the name of the column containing the list to process
        :param pos_column: Str the name of the column indicating the index of the value to extract
        :param new_col: Str the new column name
        :return:
        """
        self.df[new_col] = self.df.apply(lambda x: x[column][x[pos_column]], axis=1)

    def filter_packet_in_direction(self, column_name: str, direction_column: str, direction='O') -> str:
        """
        Extract Forward and Backward features
        :param column_name: Str
        :param direction_column: Str
        :param direction: [I:in Backward, O:out Forward]
        :return: Dataframe
        """
        if direction not in ['O', 'I']:
            raise ValueError("Invalid direction")

        new_col = column_name + '_' + direction
        self.df[new_col] = [[x for x, y in zip(row[column_name], row[direction_column]) if y == direction] for _, row in
                            self.df.iterrows()]
        return new_col

    def generate_col_by_filtering_packets(self, column_name: str, filtered_dir_column: str, client_hello_pos_col: str,
                                          nb_max_packets: int, start_from_tcp_syn: bool, directions: list) -> list:
        """
        Generate new columns by filtering packets in the given direction
        :param column_name: Str
        :param filtered_dir_column: Str
        :param client_hello_pos_col: Str
        :param nb_max_packets: Int
        :param start_from_tcp_syn: Boolean
        :param directions: [I:in Backward, O:out Forward]
        :return: List of new columns
        """
        new_col = self.select_packets_by_nb_tcp_syn(column_name, client_hello_pos_col, nb_max_packets,
                                                    start_from_tcp_syn)
        new_features = [new_col]
        for direc in directions:
            new_filter_col = self.filter_packet_in_direction(new_col, filtered_dir_column, direc)
            new_features.append(new_filter_col)

        return new_features

    def select_generate_cols_by_packets(self, config: dict):
        """
        Entails filtering the packets according to predetermined standards, such as the quantity of packets and the
        packet direction (forward or backward). The algorithm then creates new columns by extracting the values of
        specific packet feature values from the filtered packets traveling in the specified direction.
        :param config: Dictionary containing configuration parameters
        :return: Modified dataframe
        """
        nb_max_packets = config['nb_max_packets']
        start_from_tcp_syn = config['start_from_tcp_syn']
        self.df['new_tls.handshake.type'] = self.df['tls.handshake.type'].map(convert_str_list_of_list)
        get_handshake_type_index(self.df, [1, 2], 'new_tls.handshake.type')
        self.df["packet_directions"] = self.df["packet_directions"].map(convert_str_list_to_list)
        new_direction_col = self.select_packets_by_nb_tcp_syn('packet_directions', 'ht_pos1', nb_max_packets,
                                                              start_from_tcp_syn)

        for feature in ["frame.time_epoch", "ip.len", 'tcp.offset', 'ip.ttl', "tcp.window_size_value"]:
            self.df[feature] = self.df[feature].map(convert_str_list_to_list_int)
            self.get_packet_val(feature, 'ht_pos1', 'new_' + feature + '_HC')
            self.generate_col_by_filtering_packets(feature, new_direction_col, 'ht_pos1', nb_max_packets,
                                                   start_from_tcp_syn, ['O', 'I'])

        self.df["tcp.flags"] = self.df["tcp.flags"].map(convert_str_list_to_list)
        self.get_packet_val('tcp.flags', 'ht_pos1', 'new_tcp.flags_HC')
        self.generate_col_by_filtering_packets('tcp.flags', new_direction_col, 'ht_pos1', nb_max_packets,
                                               start_from_tcp_syn, ['O', 'I'])

        return self.df
