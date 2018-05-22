import pandas as pd
import numpy as np

pd.set_option('display.width', 1000)


def preprocess(df, is_test):
    print('Original data:')
    print(df.head(20))
    print('Nan count:')
    print(df.isnull().sum())

    # drop useless columns
    print("Dropping columns...")
    df.drop(['date_time', 'site_id', 'prop_brand_bool'], axis=1, inplace=True)
    if not is_test:
        df.drop(['gross_bookings_usd', 'position'], axis=1, inplace=True)

    # fill missing review score with 0 (no information available)
    print("Filling prop_review_score...")
    df['prop_review_score'].fillna(0, inplace=True)

    # fill location score2 with 0 (least desired) TODO or -1?
    print("Filling prop_location_score2...")
    df['prop_location_score2'].fillna(0, inplace=True)

    # no star rating history for customer, take the middle?
    print("Filling visitor_hist_starrating...")
    df['visitor_hist_starrating'].fillna(3, inplace=True)

    # no purchase history for customers, make it 0?
    print("Filling visitor_hist_adr_usd...")
    df['visitor_hist_adr_usd'].fillna(0, inplace=True)

    # TODO no idea what this is
    print("Filling srch_query_affinity_score...")
    df['srch_query_affinity_score'].fillna(0, inplace=True)

    # don't know distance, users like to know the distance
    print("Filling orig_destination_distance...")
    df['orig_destination_distance'].fillna(-1, inplace=True)

    drop_comp(df)

    create_price_order(df)

    normalize(df, "price_usd")
    normalize(df, "prop_location_score2")
    normalize(df, "prop_location_score1")

    if not is_test:
        create_target_score(df)

    df.sort_values(by='srch_id', inplace=True)

    print('-' * 80)
    print('Final data:')
    print(df.head(20))
    print('Nan count:')
    print(df.isnull().sum())
    print(df.info())


def drop_comp(df):
    print("Creating comp...")
    # fill in competitor values with 0 (no difference with competitors)

    # df["comp"] = 0
    #
    # df.loc[(df['comp1_rate'] == -1) | (df['comp2_rate'] == -1) | (df['comp3_rate'] == -1) | (df['comp4_rate'] == -1) |
    #        (df['comp5_rate'] == -1) | (df['comp6_rate'] == -1) | (df['comp7_rate'] == -1) | (df['comp8_rate'] == -1),
    #        "comp"] = 1
    #
    # df.loc[(df['comp1_inv'] == 0) | (df['comp2_inv'] == 0) | (df['comp3_inv'] == 0) | (df['comp4_inv'] == 0) |
    #        (df['comp5_inv'] == 0) | (df['comp6_inv'] == 0) | (df['comp7_inv'] == 0) | (df['comp8_inv'] == 0),
    #        "comp"] += 1

    for i in range(1, 9):
        rate_col = 'comp' + str(i) + '_rate'
        inv_col = 'comp' + str(i) + '_inv'
        rate_percent_diff_col = 'comp' + str(i) + '_rate_percent_diff'

        print("Dropping " + rate_col + "...")
        df.drop(rate_col, axis=1, inplace=True)

        print("Dropping " + inv_col + "...")
        df.drop(inv_col, axis=1, inplace=True)

        print("Dropping " + rate_percent_diff_col + "...")
        df.drop(rate_percent_diff_col, axis=1, inplace=True)


def create_price_order(df):
    print("Creating price_order...")
    df["price_order"] = -1

    df.sort_values(["srch_id", "price_usd"], inplace=True, ascending=[True, True])

    i = 0
    curr_id = -1
    for index, row in df.iterrows():
        if row["srch_id"] != curr_id:
            curr_id = row["srch_id"]
            i = 0

        df.at[index, "price_order"] = i
        i += 1


def normalize(df, column_name):
    print("Normalizing " + column_name + "...")
    df[column_name] = (df[column_name] - df[column_name].mean()) / df[column_name].std()


def create_target_score(df):
    print("Creating target_score...")
    df["target_score"] = 0
    df.loc[df["click_bool"] == 1, "target_score"] = 1
    df.loc[df["booking_bool"] == 1, "target_score"] = 5

    df.drop(["click_bool", "booking_bool"], axis=1, inplace=True)


if __name__ == '__main__':
    import time
    import sys

    if len(sys.argv) <= 3:
        print("Usage: [input_file] [output_file] [is_test_set: 0 or 1]")
        exit(0)

    start_ts = time.time()

    in_file = sys.argv[1]
    out_file = sys.argv[2]
    is_test = int(sys.argv[3]) == 1

    print("Reading " + in_file + "...")
    df = pd.read_csv(in_file)

    preprocess(df, is_test)

    print("Writing " + out_file + "...")
    df.to_csv(out_file, index=False)

    print("Finished in " + str(time.time() - start_ts) + " seconds.")
