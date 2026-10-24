import re
from collections import Counter

t = open('README.md', encoding='utf-8').read()


def slug(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r'[^\w\s\-]', '', s, flags=re.UNICODE)
    s = re.sub(r'\s+', '-', s)
    return s


heads = []
for line in t.splitlines():
    m = re.match(r'^(#{1,6})\s+(.*)$', line)
    if m:
        heads.append(m.group(2).strip())
counts = Counter()
anchors = set()
for h in heads:
    a = slug(h)
    if a in anchors:
        counts[a] += 1
        anchors.add(f'{a}-{counts[a]}')
    else:
        anchors.add(a)

links = re.findall(r'\]\(#([^)]+)\)', t)
missing = []
for l in links:
    if l not in anchors:
        missing.append(l)
print('headings', len(heads), 'anchors', len(anchors), 'internal links', len(links))
print('missing', missing)
print('sample anchors', sorted(anchors)[:8])
