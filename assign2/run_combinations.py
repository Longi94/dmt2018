from train import train
from ndcg import calculate_ndcg
from predict import predict

import pandas as pd

import sys
import time
import itertools
import math


def ncr(n, r):
    f = math.factorial
    return f(n) // f(r) // f(n - r)


start_ts = time.time()

in_train_file = sys.argv[1]
in_test_file = sys.argv[2]

print("Reading " + in_train_file + "...")
df_train = pd.read_csv(in_train_file)

print("Reading " + in_test_file + "...")
df_test = pd.read_csv(in_test_file)

columns = df_train.columns.tolist()
columns.remove("srch_id")
columns.remove("prop_id")
columns.remove("target_score")

results = []

for L in range(len(columns) - 3, len(columns) + 1):
    i = 0
    subsets = ncr(len(columns), L)
    for subset in itertools.combinations(columns, L):
        i += 1
        print("Length: {}, Subset {}/{}".format(L, i, subsets))
        print(subset)
        sub_df_train = df_train[list(subset) + ["srch_id", "prop_id", "target_score"]]
        sub_df_test = df_test[list(subset)]
        model = train(sub_df_train, 1)

        df_prediction = pd.DataFrame({
            "SearchId": df_test["srch_id"],
            "PropertyId": df_test["prop_id"],
            "result": model.predict(sub_df_test)
        })

        ndcg = calculate_ndcg(df_test, df_prediction)

        result = {
            "nDCG": ndcg
        }

        importances = zip(sub_df_train.columns.values, list(model.feature_importances_))

        for feature in importances:
            result[feature[0]] = feature[1]

        results.append(result)

pd.DataFrame(results).to_csv("stats.csv", index=False)

print("Finished in " + str(time.time() - start_ts) + " seconds.")
