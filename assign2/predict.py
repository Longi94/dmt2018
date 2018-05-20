import pandas as pd


def predict(model, df_test):
    x_test = df_test.drop(["srch_id", "prop_id"], axis=1)

    if "click_bool" in x_test:
        x_test.drop(["click_bool"], axis=1, inplace=True)

    if "booking_bool" in x_test:
        x_test.drop(["booking_bool"], axis=1, inplace=True)

    print("Predicting...")
    result = model.predict(x_test)

    return pd.DataFrame({
        "SearchId": df_test["srch_id"],
        "PropertyId": df_test["prop_id"],
        "result": result
    })


if __name__ == '__main__':
    import pickle
    import sys
    import time

    if len(sys.argv) <= 3:
        print("""Usage: [input_model_file] [input_test_file] [output_file]""")
        exit(0)

    start_ts = time.time()

    in_model_file = sys.argv[1]
    in_test_file = sys.argv[2]
    out_file = sys.argv[3]

    print("Loading model " + in_model_file + "...")
    model = pickle.load(open(in_model_file, 'rb'))

    print("Reading " + in_test_file + "...")
    df_test = pd.read_csv(in_test_file)

    submission = predict(model, df_test)

    print("Writing " + out_file + "...")
    submission.to_csv(out_file, index=False)

    print("Finished in " + str(time.time() - start_ts) + " seconds.")
