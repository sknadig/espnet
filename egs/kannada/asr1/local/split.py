import sys
import pandas as pd

in_file=sys.argv[1]
out_file=sys.argv[2]

df = pd.read_csv(in_file, sep='\t', header=None)
data = df.values

with open(out_file, 'w') as f:
    for row in data:
        t = ' '.join([char for char in ' '.join(row[1:])])
        if any(char.isdigit() for char in t):
            f.write(f'{row[0]} {t}\n')