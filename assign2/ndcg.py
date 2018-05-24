import pandas as pd
from pyltr.metrics import NDCG


def calculate_ndcg(truth, prediction):
    print("Calculating score...")
    return NDCG(k=len(truth)).calc_mean(truth["srch_id"].values, truth["target_score"].values, prediction["result"].values)


if __name__ == '__main__':
    import sys
    import time

    if len(sys.argv) <= 2:
        print("""Usage: [truth file] [prediction file]""")
        exit(0)

    start_ts = time.time()

    truth_file = sys.argv[1]
    prediction_file = sys.argv[2]

    print("Reading " + truth_file + "...")
    df_truth = pd.read_csv(truth_file)

    print("Reading " + prediction_file + "...")
    df_prediction = pd.read_csv(prediction_file)

    score = calculate_ndcg(df_truth, df_prediction)

    print(score)

    print("Finished in " + str(time.time() - start_ts) + " seconds.")
