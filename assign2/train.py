import time
import sys
import pandas as pd
import numpy as np

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from pyltr.models.lambdamart import LambdaMART
from ndcg import ndcg_score


def main():
    if len(sys.argv) <= 3:
        print("Usage: [input_training_file] [input_test_file] [output_file]")
        return

    in_train_file = sys.argv[1]
    in_test_file = sys.argv[2]
    out_file = sys.argv[3]

    print("Reading " + in_train_file + "...")
    df_train = pd.read_csv(in_train_file)

    # x_train = df_train.drop(["result", "srch_id", "prop_id"], axis=1)
    # y_train = df_train["result"]
    x_train = df_train.drop(["click_bool", "booking_bool", "srch_id", "prop_id"], axis=1)
    y_train = df_train["booking_bool"].copy()

    del df_train

    print("Fitting Linear regression...")
    alg = GradientBoostingClassifier()
    alg.fit(x_train, y_train)

    print("Reading " + in_test_file + "...")
    df_test = pd.read_csv(in_test_file)

    x_test = df_test.drop(["srch_id", "prop_id"], axis=1)
    print("Predicting...")
    model = alg.predict(x_test)

    print("Calculating cross validation score...")
    score = cross_val_score(alg, x_train, y_train)
    print("Cross validation score: " + str(score))

    submission = pd.DataFrame({
        "SearchId": df_test["srch_id"],
        "PropertyId": df_test["prop_id"],
        "result": model
    })

    print("Writing " + out_file + "...")
    submission.to_csv(out_file, index=False)


if __name__ == '__main__':
    start_ts = time.time()
    main()
    print("Finished in " + str(time.time() - start_ts) + " seconds.")
