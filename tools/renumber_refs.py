import re
p = 'docs/10_appendix.md'
t = open(p, encoding='utf-8').read()
start = t.index('## 参考文献')
head, body = t[:start], t[start:]
n = 1
out = []
for line in body.splitlines():
    m = re.match(r'^(\d+)\. (.*)$', line)
    if m:
        out.append(f'{n}. {m.group(2)}')
        n += 1
    else:
        out.append(line)
body = '\n'.join(out)
open(p, 'w', encoding='utf-8').write(head + body)
print('renumbered', n - 1, 'references')
