import re, os
t = open('README.md', encoding='utf-8').read()
print('lines', t.count('\n') + 1, 'chars', len(t))
print('fences', t.count('\n```'), 'dollar', t.count('\n$$'))
print('TODO', t.count('TODO'), 'TBD', t.count('TBD'))
for m in re.finditer(r'!\[[^\]]*\]\(([^)]+)\)', t):
    p = m.group(1)
    print('img', p, os.path.exists(p))
for m in re.finditer(r'<img src="([^"]+)"', t):
    p = m.group(1)
    print('htmlimg', p, os.path.exists(p))
for bad in ['#付録f', '#付録e', '#付録d-', '#付録b-']:
    if bad in t:
        print('BAD LINK', bad)
# check for duplicate headings
heads = re.findall(r'^## .+$', t, re.M)
from collections import Counter
for h, c in Counter(heads).items():
    if c > 1:
        print('DUP', h, c)
print('headings', len(heads))
