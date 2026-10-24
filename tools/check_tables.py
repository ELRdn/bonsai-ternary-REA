import re

lines = open('README.md', encoding='utf-8').read().splitlines()
blocks = []
cur = []
for i, line in enumerate(lines, 1):
    if line.strip().startswith('|'):
        cur.append((i, line))
    else:
        if cur:
            blocks.append(cur)
            cur = []
if cur:
    blocks.append(cur)

bad = 0
for b in blocks:
    counts = []
    for i, line in b:
        # remove inline code spans so |w| inside backticks does not count
        s = re.sub(r'`[^`]*`', '', line)
        counts.append(s.count('|'))
    if len(set(counts)) > 1:
        bad += 1
        print('MISMATCH block starting line', b[0][0], 'counts', sorted(set(counts)))
        for i, line in b[:4]:
            print(' ', i, line[:160])
print('table blocks', len(blocks), 'bad', bad)
