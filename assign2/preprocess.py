import sys
import pandas as pd
import time

pd.set_option('display.width', 1000)


def main():
    if len(sys.argv) <= 3:
        print("Usage: [input_file] [output_file] [is_test_set: 0 or 1]")
        return

    in_file = sys.argv[1]
    out_file = sys.argv[2]
    is_test = int(sys.argv[3]) == 1

    print("Reading " + in_file + "...")
    df = pd.read_csv(in_file)

    print('Original data:')
    print(df.head(20))
    print('Nan count:')
    print(df.isnull().sum())

    # drop useless columns
    print("Dropping columns...")
    df.drop(['date_time', 'site_id'], axis=1, inplace=True)
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

    fill_comps(df)

    df.sort_values(by='srch_id', inplace=True)

    # if not is_test:
    #     create_result(df)

    print('-' * 80)
    print('Final data:')
    print(df.head(20))
    print('Nan count:')
    print(df.isnull().sum())
    print(df.info())

    print("Writing " + out_file + "...")
    df.to_csv(out_file, index=False)


def fill_comps(df):
    # fill in competitor values with 0 (no difference with competitors)
    for i in range(1, 9):
        rate_col = 'comp' + str(i) + '_rate'
        inv_col = 'comp' + str(i) + '_inv'
        rate_percent_diff_col = 'comp' + str(i) + '_rate_percent_diff'

        print("Filling " + rate_col + "...")
        df[rate_col].fillna(0, inplace=True)
        df[rate_col] = df[rate_col].astype(int)

        print("Filling " + inv_col + "...")
        df[inv_col].fillna(0, inplace=True)
        df[inv_col] = df[inv_col].astype(int)

        print("Filling " + rate_percent_diff_col + "...")
        df[rate_percent_diff_col].fillna(0, inplace=True)


def create_result(df):
    # Create a new column from the click_bool and the book_bool. This is what we will train to predict and order the
    # result by. If clicked then 0.5, if booked then 1, 0 otherwise
    df['result'] = 0
    df.loc[df['click_bool'] == 1, 'result'] = 0.5
    df.loc[df['booking_bool'] == 1, 'result'] = 1
    df.drop(['click_bool', 'booking_bool'], axis=1, inplace=True)


if __name__ == '__main__':
    start_ts = time.time()
    main()
    print("Finished in " + str(time.time() - start_ts) + " seconds.")
