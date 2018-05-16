import time
import sys
import pandas as pd
import pickle

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from pyltr.models.lambdamart import LambdaMART

LAMBDA_MART = 0
GBM_ENSEMBLE = 1


def main():
    if len(sys.argv) <= 3:
        print("""Usage: [input_training_file] [input_test_file] [output_file] [model type]
        0 - LambdaMART
        1 - Ensemble of Gradient Boosting Classifiers""")
        return

    in_train_file = sys.argv[1]
    out_file = sys.argv[2]
    model_type = int(sys.argv[3])

    print("Reading " + in_train_file + "...")
    df_train = pd.read_csv(in_train_file)

    if model_type == LAMBDA_MART:
        model = train_lambda_mart(df_train)
    elif model_type == GBM_ENSEMBLE:
        model = train_gbm_ensemble(df_train)
    else:
        print("Unknown model type: " + str(model_type))
        return

    print("Dumping model to " + out_file + "...")
    pickle.dump(model, open(out_file, 'wb'))


def train_lambda_mart(df_train):
    x_train = df_train.drop(["click_bool", "booking_bool", "srch_id", "prop_id"], axis=1)
    y_train = df_train["booking_bool"].copy()
    query_ids = df_train["srch_id"].copy()

    model = LambdaMART(n_estimators=100, verbose=3)
    model.fit(x_train, y_train, query_ids)

    return model


def train_gbm_ensemble(df_train):
    x_train = df_train.drop(["click_bool", "booking_bool", "srch_id", "prop_id"], axis=1)
    y_train = df_train["booking_bool"].copy()

    print("Fitting GradientBoostingClassifier...")
    model = GradientBoostingClassifier(n_estimators=100, verbose=3)
    model.fit(x_train, y_train)

    print("Calculating cross validation score...")
    score = cross_val_score(model, x_train, y_train)
    print("Cross validation score: " + str(score))

    return model


if __name__ == '__main__':
    start_ts = time.time()
    main()
    print("Finished in " + str(time.time() - start_ts) + " seconds.")
