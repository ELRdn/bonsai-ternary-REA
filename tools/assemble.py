import os, re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILES = [
    'docs/00_intro.md',
    'docs/01_basics.md',
    'docs/02_overview.md',
    'docs/03_re.md',
    'docs/03b_conclusion.md',
    'docs/04_rotation.md',
    'docs/05_kernels.md',
    'docs/06_bench.md',
    'docs/07_compare.md',
    'docs/08_deploy.md',
    'docs/09_open.md',
    'docs/10_appendix.md',
]

parts=[]
for f in FILES:
    p=os.path.join(ROOT,f)
    t=open(p,encoding='utf-8').read().strip()
    # normalize headings: file 00 already has title; others start with ##
    parts.append(t)
out='\n\n'.join(parts)+'\n'
open(os.path.join(ROOT,'README.md'),'w',encoding='utf-8').write(out)
print('README.md',len(out),'chars',out.count('\n')+1,'lines')
