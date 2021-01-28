

if __name__ == '__main__':
    print("hello word")
    # parse command line
    parser = argparse.ArgumentParser(description='''
    This script takes two data files with timestamps and associates them   
    ''')
    parser.add_argument('first_file', help='first text file (format: timestamp data)')
    # parser.add_argument('second_file', help='second text file (format: timestamp data)')
    # parser.add_argument('three_file',help='three text file (format: timestamp data)')
    # parser.add_argument('--first_only', help='only output associated lines from first file', action='store_true')
    # parser.add_argument('--offset', help='time offset added to the timestamps of the second file (default: 0.0)',default=0.0)
    # parser.add_argument('--max_difference', help='maximally allowed time difference for matching entries (default: 0.02)',default=0.02)
    args = parser.parse_args()

