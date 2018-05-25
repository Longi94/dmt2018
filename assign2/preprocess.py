import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

pd.set_option('display.width', 1000)
label = LabelEncoder()


def preprocess(df, is_test):
    print('Original data:')
    print(df.head(20))
    print('Nan count:')
    print(df.isnull().sum())

    if not is_test:
        df.drop(['gross_bookings_usd', 'position'], axis=1, inplace=True)

    remove_price_usd_outliers(df)

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

    # don't know distance, users like to know the distance, fill in with "far away"
    print("Filling orig_destination_distance...")
    normalize_distance(df)
    df['orig_destination_distance'].fillna(2, inplace=True)

    drop_comp(df)

    create_price_order(df)

    df['is_alone'] = np.logical_and(df['srch_adults_count']==1, df['srch_children_count']==0)

    df['price_diff'] = np.abs(np.log(df['price_usd'] + 1) - df['prop_log_historical_price'])

    #df['square_price_diff_window'] = (np.abs(np.log(df['price_usd'] + 1) - df['prop_log_historical_price']))*(df["srch_booking_window"]*(df["srch_booking_window"]+1))
    # normalize(df, "price_usd")
    # normalize(df, "prop_location_score2")
    # normalize(df, "prop_location_score1")

    create_loc_rank(df)

    #create_loc_rank_children(df)

    create_price_diff_trend(df)

    create_price_hurry(df)

    create_price_behavior(df)

    band(df, "price_usd", 5)

    df['quality_pricestar_ratio'] = df['prop_review_score'] / (df['price_usd'] + df['prop_starrating'] + 1)
    df['quality_price'] = df['prop_review_score'] / (df['price_usd'] + 1)
    df['quality_star'] = (df['prop_review_score'] + 1) / (df['prop_starrating'] + 1)

    df['hurry'] = (df['srch_query_affinity_score'] + 1) / (np.log(df['srch_booking_window'] + 1) + 1)

    # create_is_alone(df)

    if not is_test:
        create_target_score(df)

    # drop useless columns
    print("Dropping columns...")
    df.drop(['date_time',
             'site_id',
             'prop_brand_bool',
             'random_bool',
             'srch_destination_id',
             'prop_country_id',
             'visitor_location_country_id',
             'srch_saturday_night_bool',
             'srch_room_count',
             'hurry',
             'price_usd',
             'visitor_hist_starrating'], axis=1,
            inplace=True)

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

    df["comp"] = 0

    df.loc[((df['comp1_rate'] == -1) & (df['comp1_inv'] == 0)) |
           ((df['comp2_rate'] == -1) & (df['comp1_inv'] == 0)) |
           ((df['comp3_rate'] == -1) & (df['comp1_inv'] == 0)) |
           ((df['comp4_rate'] == -1) & (df['comp1_inv'] == 0)) |
           ((df['comp5_rate'] == -1) & (df['comp1_inv'] == 0)) |
           ((df['comp6_rate'] == -1) & (df['comp1_inv'] == 0)) |
           ((df['comp7_rate'] == -1) & (df['comp1_inv'] == 0)) |
           ((df['comp8_rate'] == -1) & (df['comp1_inv'] == 0)),
           "comp"] = 1

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

def create_price_diff(df):
    print("creating price diff")
    df['price_diff'] = 0
    df['price_diff'] = np.abs(np.log(df['price_usd'] + 1) - df['prop_log_historical_price'])


def create_quality_star(df):
    print("creating quality star ratio")
    df['quality_star'] = 0.
    df['quality_star'] = (df['prop_review_score']+1)/(df['prop_starrating']+1)

def create_quality_price(df):
    print("creating quality price ratio")
    df['quality_price'] = 0.
    df['quality_price'] = (df['prop_review_score']+1)/(df['price_range']+1)

def create_price_range(df):
    print("creating price range")
    df['price_range'] = (df['price_usd']+(200 - df['price_usd']%200))/200

def create_hurry(df):
    print("creating hurry")
    df["hurry"] = 0.
    df['hurry'] = (df['srch_query_affinity_score']+1)/(np.log(df['srch_booking_window']+1)+1)

def create_price_diff_trend(df):
    print("Creating price diff trend")
    df["diff_trend"] = 0
    df['diff_trend'] = np.sign(np.log(df['price_usd'] + 1) - df['prop_log_historical_price'])

def create_price_behavior(df):
    print("creating price behavior")
    df["price_behavior"] = 0
    df["price_behavior"] = (df["diff_trend"]*df["price_hurry"])


def create_price_hurry(df):
    print("creating price hurry")
    df["price_hurry"] = 0
    df['price_hurry'] = df["price_diff"]/(df['srch_booking_window']+1)

def create_loc_rank(df):
    print("creating location rank")
    df["loc_rank"] = (df["prop_location_score1"] + df["prop_location_score2"]) / (df["price_order"] + 1)

def create_loc_rank_children(df):
    print("creating location rank children")
    df["loc_rank_children"] = (df["loc_rank"]*df["srch_children_count"])


def remove_price_usd_outliers(df):
    q = df["price_usd"].quantile(0.98)
    df.drop(df[df["price_usd"] > q].index, inplace=True)


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


def band(df, column, count):
    band_column = column + "_band"
    df[band_column] = pd.qcut(df[column], count)
    df[column] = label.fit_transform(df[band_column])
    df.drop([band_column], axis=1, inplace=True)


def normalize_distance(df):
    print("Normalizing orig_destination_distance...")
    for srch_id in df.loc[df["orig_destination_distance"].notnull(), "srch_id"].unique():
        mean = df.loc[df["srch_id"] == srch_id, "orig_destination_distance"].mean()
        std = df.loc[df["srch_id"] == srch_id, "orig_destination_distance"].std()
        df.loc[df["srch_id"] == srch_id, "orig_destination_distance"] = (df["orig_destination_distance"] - mean) / std


def create_is_alone(df):
    print("Creating is_alone...")
    df["is_alone"] = 0
    df.loc[(df["srch_children_count"] + df["srch_adults_count"]) == 1, "is_alone"] = 1
    df.drop(["srch_children_count", "srch_adults_count"], axis=1, inplace=True)


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
