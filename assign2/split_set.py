import pandas as pd


def split_set(input_df, df_ratio):
    search_ids = input_df["srch_id"].unique()

    head_len = round(len(search_ids) * df_ratio)
    train_ids = search_ids[:head_len]
    validation_ids = search_ids[head_len:]

    train_set = input_df[input_df["srch_id"].isin(train_ids)]
    validation_set = input_df[input_df["srch_id"].isin(validation_ids)]

    return train_set, validation_set


if __name__ == '__main__':
    import sys
    import time
    import os

    if len(sys.argv) <= 3:
        print("""Usage: [input file] [output_folder] [ratio]""")
        exit(0)

    start_ts = time.time()

    input_file = sys.argv[1]
    out_folder = sys.argv[2]
    ratio = float(sys.argv[3])

    print("Reading " + input_file + "...")
    df_train = pd.read_csv(input_file)

    train_set, test_set = split_set(df_train, ratio)

    os.makedirs(out_folder, exist_ok=True)

    print("Writing " + out_folder + "/train_set.csv...")
    train_set.to_csv(out_folder + "/train_set.csv", index=False)

    print("Writing " + out_folder + "/test_set.csv...")
    test_set.to_csv(out_folder + "/test_set.csv", index=False)

    print("Finished in " + str(time.time() - start_ts) + " seconds.")
