import pandas as pd
import time
import sys

if len(sys.argv) <= 2:
    print("Usage: [input_file] [output_file]")
    exit(0)

start_ts = time.time()

in_file = sys.argv[1]
out_file = sys.argv[2]

print("Reading " + in_file + "...")
df = pd.read_csv(in_file)

df.sort_values(["SearchId", "result"], inplace=True, ascending=[True, False])
df.drop("result", axis=1, inplace=True)

print("Writing " + out_file + "...")
df.to_csv(out_file, index=False)

print("Finished in " + str(time.time() - start_ts) + " seconds.")
