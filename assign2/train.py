import pandas as pd

from sklearn.utils import resample
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from pyltr.models.lambdamart import LambdaMART
from pyltr.metrics import NDCG

LAMBDA_MART = 0
GBM_ENSEMBLE = 1


def train(df_train, model_type):
    #df_balanced = balance_data(df_train)
    df_balanced = df_train
    if model_type == LAMBDA_MART:
        model = train_lambda_mart(df_train)
    elif model_type == GBM_ENSEMBLE:
        model = train_gbm_ensemble(df_train)
    else:
        print("Unknown model type: " + str(model_type))
        return

    return model


def train_lambda_mart(df_train):
    x_train = df_train.drop(["target_score", "srch_id", "prop_id"], axis=1)
    y_train = df_train["target_score"]
    query_ids = df_train["srch_id"].copy()

    print("Fitting LambdaMART...")
    model = LambdaMART(metric=NDCG(len(df_train)), n_estimators=100, verbose=1)
    model.fit(x_train, y_train, query_ids)

    print_feature_importances(x_train, model)

    return model


def train_gbm_ensemble(df_train):
    x_train = df_train.drop(["target_score", "srch_id", "prop_id"], axis=1)
    y_train = df_train["target_score"]

    print("Fitting GradientBoostingClassifier...")
    model = GradientBoostingRegressor(n_estimators=100, verbose=1)
    model.fit(x_train, y_train)

    print_feature_importances(x_train, model)

    return model


def print_feature_importances(x_train, model):
    print("Important features")
    importances = zip(x_train.columns.values, list(model.feature_importances_))
    features = [feature for feature in importances]
    features.sort(key=lambda x: x[1], reverse=True)
    for feature in features:
        print(feature)


def balance_data(df):
    print("Balancing data...")
    # Separate majority and minority classes
    n_upsample = df.loc[df['target_score'] > 0].shape[0]
    df_majority = df.loc[df['target_score'] == 0]
    df_minority = df.loc[df['target_score'] > 0]

    # Upsample minority class
    df_majority_downsampled = resample(df_majority,
                                     replace=True,  # sample with replacement
                                     n_samples=2*n_upsample,  # to match majority class
                                     random_state=123)  # reproducible results

    # Combine majority class with upsampled minority class
    balanced = pd.concat([df_minority, df_majority_downsampled])
    balanced.sort_values(by='srch_id', inplace=True)
    return balanced


if __name__ == '__main__':
    import time
    import sys
    import pickle

    if len(sys.argv) <= 3:
        print("""Usage: [input_training_file] [output_file] [model type]
        0 - LambdaMART
        1 - Ensemble of Gradient Boosting Classifiers""")
        exit(0)

    start_ts = time.time()

    in_train_file = sys.argv[1]
    out_file = sys.argv[2]
    model_type = int(sys.argv[3])

    if model_type > 1 or model_type < 0:
        print("Unknown model type: " + str(model_type))
        exit(0)

    print("Reading " + in_train_file + "...")
    df_train = pd.read_csv(in_train_file)

    model = train(df_train, model_type)

    print("Dumping model to " + out_file + "...")
    pickle.dump(model, open(out_file, 'wb'))

    print("Finished in " + str(time.time() - start_ts) + " seconds.")
