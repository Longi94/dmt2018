import sys
import pandas as pd

pd.set_option('display.width', 1000)


def main():
    if len(sys.argv) < 2:
        print("Usage: [input_file] [output_file]")
        return

    in_file = sys.argv[1]
    out_file = sys.argv[2]

    df = pd.read_csv(in_file)

    print('Original data:')
    print(df.head(20))
    print('Nan count:')
    print(df.isnull().sum())

    # drop useless columns
    df.drop(['date_time'], axis=1, inplace=True)

    # fill missing review score with 0 (no information available)
    df['prop_review_score'].fillna(0, inplace=True)

    # fill location score2 with 0 (least desired) TODO or -1?
    df['prop_location_score2'].fillna(0, inplace=True)

    # no star rating history for customer, take the middle?
    df['visitor_hist_starrating'].fillna(3, inplace=True)

    # no purchase history for customers, make it 0?
    df['visitor_hist_adr_usd'].fillna(0, inplace=True)

    # TODO no idea what this is
    df['srch_query_affinity_score'].fillna(0, inplace=True)

    # don't know distance, users like to know the distance
    df['orig_destination_distance'].fillna(-1, inplace=True)

    # no transaction
    df['gross_bookings_usd'].fillna(0, inplace=True)

    fill_comps(df)

    print('-' * 80)
    print('Final data:')
    print(df.head(20))
    print('Nan count:')
    print(df.isnull().sum())
    print(df.info())

    df.to_csv(out_file, index=False)


def fill_comps(df):
    # fill in competitor values with 0 (no difference with competitors)
    for i in range(1, 9):
        df['comp' + str(i) + '_rate'].fillna(0, inplace=True)
        df['comp' + str(i) + '_rate'] = df['comp' + str(i) + '_rate'].astype(int)
        df['comp' + str(i) + '_inv'].fillna(0, inplace=True)
        df['comp' + str(i) + '_inv'] = df['comp' + str(i) + '_inv'].astype(int)
        df['comp' + str(i) + '_rate_percent_diff'].fillna(0, inplace=True)


if __name__ == '__main__':
    main()
