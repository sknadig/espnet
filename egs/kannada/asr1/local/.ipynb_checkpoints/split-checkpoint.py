import sys

in_file=sys.argv[0]
out_file=sys.argv[1]

data = open(in_file).readlines()
data = [row.strip().split('\t') for row in data]

with open(out_file, 'w') as f:
    for row in data:
        t = ' '.join([char for char in row[1]])
        f.write(f'{row[0]} {t}\n')