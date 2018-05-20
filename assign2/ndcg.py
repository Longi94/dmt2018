from math import log
import pandas as pd
import numpy as np


def dcg_at_k(scores):
    return scores[0] + sum(sc / log(ind, 2) for sc, ind in zip(scores[1:], range(2, len(scores) + 1)))


def ndcg_at_k(predicted_scores, user_scores):
    assert len(predicted_scores) == len(user_scores)
    idcg = dcg_at_k(sorted(user_scores, reverse=True))
    return (dcg_at_k(predicted_scores) / idcg) if idcg > 0.0 else 0.0


def calculate_ndcg(truth, prediction):
    print("Calculating score...")
    scores = []

    for srch_id in truth["srch_id"].unique():
        y_true = truth.loc[truth["srch_id"] == srch_id]
        y_true = y_true["booking_bool"] + y_true["click_bool"]

        y_score = prediction.loc[prediction["SearchId"] == srch_id]["result"]

        score = ndcg_at_k(y_true.tolist(), y_score.tolist())
        scores.append((srch_id, score))
        print((srch_id, score))

    return np.mean([score[1] for score in scores])


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
