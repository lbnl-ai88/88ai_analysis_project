import pandas as pd
import os


def read_column_names(column_names_file):
    column_names = {}
    with open(column_names_file, 'r') as file:
        for line in file:
            index, name = line.strip().split()
            column_names[int(index)] = name
    return column_names


def read_watch_data(file_list, column_names):
    all_data = []

    for file_path in file_list:
        run_id = os.path.splitext(os.path.basename(file_path))[0].split('_')[-1]
        data = []
        accumulation_group = 0
        counter = 1
        with open(file_path, 'r') as file:
            for line in file:
                values = line.strip().split()
                row = {'accumulation_group': accumulation_group}
                if counter % 60 == 0:
                    accumulation_group += 1
                counter += 1
                for i in range(1, len(values)):
                    row[column_names[i - 1]] = float(values[i])
                row['run_id'] = run_id
                data.append(row)
        df = pd.DataFrame(data)
        all_data.append(df)
    combined_df = pd.concat(all_data, ignore_index=True)
    return combined_df


def save_to_parquet(df, output_file):
    df.to_parquet(output_file, engine='pyarrow')


def main():
    column_names_file = '../data/raw_text/column_names'
    file_list = ['../data/raw_text/watch_data_' + run for run in ['00', '01', '02', '03', '04', '05', '06', '08', '08a', '08b', '08c', '08d', '09', '10', '10b', '11', '11b', '12', '13', '14']]  # Add all your files here
    output_file = '../data/raw_watch_data.parquet'

    column_names = read_column_names(column_names_file)
    combined_df = read_watch_data(file_list, column_names)
    save_to_parquet(combined_df, output_file)


if __name__ == '__main__':
    main()
