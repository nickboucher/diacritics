#!/usr/bin/env python3
import pickle
import json
from glob import glob
from sys import argv
from datetime import datetime

output = {}

for filename in glob("*.pkl"):
    print(f"Processing {filename}...")

    with open(filename, 'rb') as f:
        for key, value in pickle.load(f).items():
            if key in output:
                output[key].update(value)
            else:
                output[key] = value

if len(argv) == 2:
    outfilename = argv[1]
else:
    outfilename = f"results-{datetime.now().strftime('%m-%d-%Y-%H:%M:%S')}"

picklefile = f'../results/{outfilename}.pkl'
print(f"Exporting Pickle as {picklefile}.")
with open(picklefile, 'wb') as f:
    pickle.dump(output, f)

jsonfile = f'../results/{outfilename}.json'
print(f"Exporting JSON as {jsonfile}.")
with open(jsonfile, 'w') as f:
    json.dump(output, f)