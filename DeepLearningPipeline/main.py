import pandas as pd

from processing import *


def main():
    config = load_config('config.yaml')

    df = get_data('C:\\data\\synth_browsers\\', config['list_of_features'])
    df.reset_index(drop=True, inplace=True)
    print(df.shape, df.uan.value_counts())
    df = format_features(df)
    df = filter_data(df, config['start_from_hc'], 20)
    df = process_tcp_ip_features(df)

    df = process_handshake_type_ext(df, config)
    cols = [col for col in df.columns if (('stat' in col) | ('tls.bit' in col))]
    cols += ['all', 'timeseries', 'tls.cipher', 'packet_directions', 'ip.ttl', 'ip.len', 'tcp.window_size_value', 'tcp.offset',
             'frame.time_epoch', 'osn', 'osf', 'uav', 'uan']
    df[cols].to_pickle('data.pkl')
    df = sample_from_categories(df[cols], 'uan', 50000)
    df[cols].to_pickle('sample.pkl')

    exit(0)


if __name__ == '__main__':
    main()
