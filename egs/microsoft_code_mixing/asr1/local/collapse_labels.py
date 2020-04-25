#!/usr/bin/env python3

import sys
import pandas as pd
import numpy as np

infile = sys.argv[1]
outfile = sys.argv[2]

df = pd.read_csv(infile, sep='\t', header=None, dtype={0: object})
data = df.values
new_data = []
for row in data:
    collapsed = [set(ele).pop() for ele in row[1].split()]
    new_data.append(' S '.join(collapsed))

new_data = np.array(new_data)
df[1] = new_data
df.to_csv(outfile, sep='\t', index=False, header=False)