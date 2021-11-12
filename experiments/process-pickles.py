#!/usr/bin/env python3
import pickle
import json
import numpy as np
from glob import glob
from sys import argv
from datetime import datetime

def denumpify(target):
  for key, value in list(target.items()):
    if isinstance(value, dict):
      denumpify(value)
    elif isinstance(value, np.ndarray):
      target[key] = value.tolist()
    elif isinstance(value, np.float32) or isinstance(value, np.bool_):
      target[key] = value.item()

output = {}

for filename in glob("*.pkl"):
    print(f"Processing {filename}...")

    with open(filename, 'rb') as f:
        for exp_name, exp in pickle.load(f).items():
            if exp_name in output:
                for budget, adv_examples in exp.items():
                    if budget in output[exp_name]:
                        output[exp_name][budget].update(adv_examples)
                    else:
                        output[exp_name][budget] = adv_examples
            else:
                output[exp_name] = exp

if len(argv) == 2:
    outfilename = argv[1]
else:
    outfilename = f"results-{datetime.now().strftime('%m-%d-%Y-%H:%M:%S')}"

# Remove numpy-specific formats
denumpify(output)

picklefile = f'../results/{outfilename}.pkl'
print(f"Exporting Pickle as {picklefile}.")
with open(picklefile, 'wb') as f:
    pickle.dump(output, f)

jsonfile = f'../results/{outfilename}.json'
print(f"Exporting JSON as {jsonfile}.")
with open(jsonfile, 'w') as f:
    json.dump(output, f)