import pickle
import sys
import pandas as pd
import time


def main():
    if len(sys.argv) <= 3:
        print("""Usage: [input_model_file] [input_test_file] [output_file]""")
        return

    in_model_file = sys.argv[1]
    in_test_file = sys.argv[2]
    out_file = sys.argv[3]

    print("Loading model " + in_model_file + "...")
    model = pickle.load(open(in_model_file, 'rb'))

    print("Reading " + in_test_file + "...")
    df_test = pd.read_csv(in_test_file)

    x_test = df_test.drop(["srch_id", "prop_id"], axis=1)
    print("Predicting...")
    result = model.predict(x_test)

    submission = pd.DataFrame({
        "SearchId": df_test["srch_id"],
        "PropertyId": df_test["prop_id"],
        "result": result
    })

    print("Writing " + out_file + "...")
    submission.to_csv(out_file, index=False)


if __name__ == '__main__':
    start_ts = time.time()
    main()
    print("Finished in " + str(time.time() - start_ts) + " seconds.")
