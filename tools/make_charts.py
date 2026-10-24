import json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager as fm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = json.load(open(os.path.join(ROOT, 'data', 'measurements.json')))
OUT = os.path.join(ROOT, 'assets')
os.makedirs(OUT, exist_ok=True)

# font
for name in ['Yu Gothic', 'Yu Gothic UI', 'Meiryo', 'Noto Sans CJK JP', 'DejaVu Sans']:
    if any(f.name == name for f in fm.fontManager.ttflist):
        plt.rcParams['font.family'] = name
        break
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 160
plt.rcParams['savefig.dpi'] = 160

NAVY='#0f172a'; TEAL='#0d9488'; ORANGE='#ea580c'; GRAY='#94a3b8'; LIGHT='#e2e8f0'; RED='#dc2626'

def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('wrote', name)

# 1 bpw
def chart_bpw():
    labels=['FP16\n(基準)','Ternary\n理想','PTQ1_0\n(27B)','PQ2_0\n(27B)','Q2_0 g64\n(27B)','MLX 2bit\n(27B)','IQ2_XXS\n(参考)']
    vals=[16.0,1.71,1.75,2.125,2.25,2.25,2.2]
    colors=[GRAY,TEAL,TEAL,ORANGE,ORANGE,'#7c3aed',GRAY]
    fig,ax=plt.subplots(figsize=(7.4,3.5))
    y=np.arange(len(labels))[::-1]
    ax.barh(y,vals,color=colors,height=.62)
    for yy,v in zip(y,vals):
        ax.text(v+0.25,yy,f'{v:.3f} bpw',va='center',fontsize=9,color=NAVY)
    ax.set_yticks(y); ax.set_yticklabels(labels,fontsize=9)
    ax.set_xlim(0,17.5); ax.set_xlabel('コーデックの実効ビット幅 (bits / weight)',fontsize=9)
    ax.set_title('三値コーデックの実効ビット幅（スケール込み）',fontsize=12,color=NAVY,loc='left')
    ax.grid(axis='x',color=LIGHT,linewidth=.8); ax.set_axisbelow(True)
    for s in ['top','right']: ax.spines[s].set_visible(False)
    save(fig,'bpw.png')

# 2 zero counts
def chart_zero_counts():
    fig,axs=plt.subplots(1,2,figsize=(7.8,3.2),sharey=True)
    for ax,key,title,color in [(axs[0],'sample_27b_gate','27B FFN gate',TEAL),(axs[1],'sample_27b_qkv','27B attn QKV',ORANGE)]:
        h=DATA[key]['zero_count_hist']
        xs=sorted(int(k) for k in h); ys=[h[str(x)] for x in xs]
        ax.bar(xs,ys,color=color,width=.7)
        ax.set_yscale('log'); ax.set_xlabel('128重みグループ中のゼロ個数',fontsize=8)
        ax.set_title(title,fontsize=9,color=NAVY)
        ax.grid(axis='y',color=LIGHT,linewidth=.8); ax.set_axisbelow(True)
        for s in ['top','right']: ax.spines[s].set_visible(False)
    axs[0].set_ylabel('グループ数 (log)',fontsize=8)
    fig.suptitle('ゼロ個数は42に集中 — 固定k設計の証拠',fontsize=11,color=NAVY,x=.02,ha='left')
    save(fig,'zero_counts.png')

# 3 zero curve
def chart_zero_curve():
    fig,ax=plt.subplots(figsize=(6.6,3.8))
    for key,color,label in [('sample_27b_gate',TEAL,'27B FFN gate'),('sample_27b_qkv',ORANGE,'27B attn QKV')]:
        bins=DATA[key]['ratio_bins']
        x=[(b['lo']+b['hi'])/2 for b in bins]; y=[b['nonzero_rate'] for b in bins]
        ax.plot(x,y,marker='o',color=color,label=label,linewidth=2)
    ax.axvline(.5,color=NAVY,linestyle='--',linewidth=1)
    ax.text(.51,.05,'通常の丸めの境界 |w|/s = 0.5',fontsize=8,color=NAVY)
    ax.set_xlabel('グループ内スケールで正規化した重みの大きさ |w| / s',fontsize=9)
    ax.set_ylabel('非ゼロ（±1）になる割合',fontsize=9)
    ax.set_title('ゼロ判定は「閾値」ではない — 滑らかに変化する',fontsize=11,color=NAVY,loc='left')
    ax.set_ylim(0,1.02); ax.grid(color=LIGHT,linewidth=.8); ax.set_axisbelow(True)
    ax.legend(frameon=False,fontsize=8,loc='lower right')
    for s in ['top','right']: ax.spines[s].set_visible(False)
    save(fig,'zero_curve.png')

# 4 sign match
def chart_sign_match():
    fig,ax=plt.subplots(figsize=(6.2,3.4))
    for key,color,label in [('sample_27b_gate',TEAL,'27B FFN gate'),('sample_27b_qkv',ORANGE,'27B attn QKV')]:
        sm=DATA[key]['sign_by_magnitude']
        x=[s['quantile']*100 for s in sm]; y=[s['sign_match']*100 for s in sm]
        ax.plot(x,y,marker='o',color=color,label=label,linewidth=2)
    ax.set_xlabel('重みの大きさのパーセンタイル（0=小さい, 75=上位25%）',fontsize=9)
    ax.set_ylabel('符号一致率 (%)',fontsize=9)
    ax.set_ylim(55,102); ax.grid(color=LIGHT,linewidth=.8); ax.set_axisbelow(True)
    ax.set_title('大きい重みの符号はほぼ完全に保存される',fontsize=11,color=NAVY,loc='left')
    ax.legend(frameon=False,fontsize=8,loc='lower right')
    for s in ['top','right']: ax.spines[s].set_visible(False)
    save(fig,'sign_match.png')

# 5 category retention
def chart_categories():
    cats=['Agentic','Coding','Math','Knowledge','Instruction','Vision','All 20']
    q=[79.74,82.17,97.06,86.66,81.25,81.64,85.354]
    b=[77.57,81.58,96.57,83.95,82.66,78.59,83.8935]
    ret=[b[i]/q[i]*100 for i in range(len(cats))]
    fig,ax=plt.subplots(figsize=(7.2,3.4))
    bars=ax.bar(cats,ret,color=[TEAL]*6+[NAVY],width=.58)
    for bar,r in zip(bars,ret):
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+1,f'{r:.1f}%',ha='center',fontsize=9,color=NAVY)
    ax.set_ylim(90,104); ax.set_ylabel('FP16比の保持率 (%)',fontsize=9)
    ax.set_title('「98.2%」の内訳 — 平均は一部の弱点を隠す',fontsize=11,color=NAVY,loc='left')
    ax.grid(axis='y',color=LIGHT,linewidth=.8); ax.set_axisbelow(True)
    for s in ['top','right']: ax.spines[s].set_visible(False)
    save(fig,'category_retention.png')

# 6 rotation
def chart_rotation():
    d=DATA['sample_27b_gate']
    fig,axs=plt.subplots(1,3,figsize=(8.2,3.0))
    vals=[[d['kurtosis_unrot'],d['kurtosis_rot']],[d['maxmean_unrot'],d['maxmean_rot']],[d['rel_mse_unrot'],d['rel_mse_rot']]]
    titles=['尖度 (kurtosis)','最大値 / 平均値','相対MSE（回転前/後）']
    for ax,v,t in zip(axs,vals,titles):
        ax.bar(['回転前','回転後'],v,color=[GRAY,TEAL],width=.55)
        for i,x in enumerate(v): ax.text(i,x*1.02 if x>0 else x, f'{x:.3f}',ha='center',fontsize=8,color=NAVY)
        ax.set_title(t,fontsize=9,color=NAVY)
        ax.grid(axis='y',color=LIGHT,linewidth=.8); ax.set_axisbelow(True)
        for s in ['top','right']: ax.spines[s].set_visible(False)
    fig.suptitle('Hadamard回転の効果：分布は正規化するが、重みMSEの改善は小さい',fontsize=10,color=NAVY,x=.02,ha='left')
    save(fig,'rotation.png')

# 7 pareto
def chart_pareto():
    fig,ax=plt.subplots(figsize=(6.4,4.0))
    pts=[('Qwen3.8-27B FP16',53.8,85.4,GRAY),('IQ2_XXS',7.3,75.2,GRAY),('Bonsai 2 27B',5.93,83.9,TEAL),('Qwen3.6-27B FP16',53.8,83.6,LIGHT)]
    for name,size,score,c in pts:
        ax.scatter(size,score,s=120,color=c,edgecolor=NAVY,linewidth=.6,zorder=3)
        ax.annotate(name,(size,score),textcoords='offset points',xytext=(7,6),fontsize=8,color=NAVY)
    ax.set_xscale('log'); ax.set_xlabel('言語モデルのサイズ (GB, log)',fontsize=9); ax.set_ylabel('20ベンチマーク平均 (%)',fontsize=9)
    ax.set_title('サイズと性能のパレート前沿',fontsize=11,color=NAVY,loc='left')
    ax.grid(color=LIGHT,linewidth=.8); ax.set_axisbelow(True)
    for s in ['top','right']: ax.spines[s].set_visible(False)
    save(fig,'pareto.png')

chart_bpw(); chart_zero_counts(); chart_zero_curve(); chart_sign_match(); chart_categories(); chart_rotation(); chart_pareto()
