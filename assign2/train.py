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
        print("""Usage: [input_training_file] [output_file] [model type]
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
    y_train = df_train["booking_bool"] + df_train["click_bool"]
    query_ids = df_train["srch_id"].copy()

    print("Fitting LambdaMART...")
    model = LambdaMART(n_estimators=100, verbose=1)
    model.fit(x_train, y_train, query_ids)

    print_feature_importances(x_train, model)

    return model


def train_gbm_ensemble(df_train):
    x_train = df_train.drop(["click_bool", "booking_bool", "srch_id", "prop_id"], axis=1)
    y_train = df_train["booking_bool"] + df_train["click_bool"]

    print("Fitting GradientBoostingClassifier...")
    model = GradientBoostingClassifier(n_estimators=100, verbose=1)
    model.fit(x_train, y_train)

    print("Calculating cross validation score...")
    score = cross_val_score(model, x_train, y_train)
    print("Cross validation score: " + str(score))

    print_feature_importances(x_train, model)

    return model


def print_feature_importances(x_train, model):
    print("Important features")
    importances = zip(x_train.columns.values, list(model.feature_importances_))
    features = [feature for feature in importances]
    features.sort(key=lambda x: x[1], reverse=True)
    for feature in features:
        print(feature)


if __name__ == '__main__':
    start_ts = time.time()
    main()
    print("Finished in " + str(time.time() - start_ts) + " seconds.")
