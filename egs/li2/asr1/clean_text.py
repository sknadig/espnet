import sys
import re

in_text = sys.argv[1]
out_text = sys.argv[2]

with open(in_text, 'r', encoding='utf-8') as f:
    lines = f.readlines()

lines = [ele.strip().split() for ele in lines]

with open(out_text, 'w', encoding='utf-8') as f:
    for line in lines:
        line_str = ' '.join(line[1:]).lower()
        if not any(char.isdigit() for char in line_str):
            line_str = re.sub(r'[^a-zA-ZÀ-ÿ0-9 \'\’]', '', line_str)
            line_str = re.sub(' +', ' ', line_str)
            f.write(line[0] + ' ' + line_str + '\n')
