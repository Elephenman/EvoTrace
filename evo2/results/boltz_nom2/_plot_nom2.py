# -*- coding: utf-8 -*-
"""nom2 双背景 sep 热图 (含显著星标 + act_S1 保活提示) + Δsep vs actS1 决策散点"""
import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

piv=pd.read_csv('cells_nom2.csv')
F88o=['F','K','R','W','Y']; M255o=['M','I','K','A']
native={'WT':'F','TrackF_r1':'K'}
n=100
def sed(p1,p2): return np.sqrt(p1*(1-p1)/n+p2*(1-p2)/n)

fig,axes=plt.subplots(1,2,figsize=(13.5,5.6))
for ax,bg in zip(axes,['WT','TrackF_r1']):
    sub=piv[piv.bg==bg]
    nb=sub[(sub.F88==native[bg])&(sub.M255=='M')].iloc[0]
    Z=np.full((len(F88o),len(M255o)),np.nan); ON=np.zeros_like(Z)
    sig=np.zeros_like(Z,dtype=int)  # 0 no,1 up, -1 down
    for i,f in enumerate(F88o):
        for j,m in enumerate(M255o):
            r=sub[(sub.F88==f)&(sub.M255==m)]
            if len(r)==0: continue
            r=r.iloc[0]; Z[i,j]=r.sep; ON[i,j]=r.act_S1
            d=r.sep-nb.sep; se=sed(r.act_S1,r.act_OFF)
            if d/se>1.96: sig[i,j]=1
            elif d/se<-1.96: sig[i,j]=-1
    im=ax.imshow(Z,cmap='RdYlGn',vmin=0.05,vmax=0.55,aspect='auto')
    ax.set_xticks(range(len(M255o))); ax.set_xticklabels([f'M{m}' for m in M255o],fontsize=11)
    ax.set_yticks(range(len(F88o))); ax.set_yticklabels([f'F{f}' for f in F88o],fontsize=11)
    ax.set_title(f'bg={bg} (native F88{native[bg]}_M255M boxed)',fontsize=12)
    ni=F88o.index(native[bg]); nj=M255o.index('M')
    ax.add_patch(plt.Rectangle((nj-0.5,ni-0.5),1,1,fill=False,edgecolor='k',lw=2.6))
    for i,f in enumerate(F88o):
        for j,m in enumerate(M255o):
            if np.isnan(Z[i,j]): continue
            star='*' if sig[i,j]==1 else ('v' if sig[i,j]==-1 else '')
            # sep 主值; actS1 若显著偏低用红, 否则黑
            on=ON[i,j]
            col='black'
            if on<0.70: col='red'
            ax.text(j,i,f'{Z[i,j]:.2f}{star}',ha='center',va='center',fontsize=10.5,
                    color=col,fontweight='bold')
    fig.colorbar(im,ax=ax,label='sep (act_S1 - act_OFF)',shrink=0.8)
plt.tight_layout()
plt.savefig('nom2_sep_heatmap.png',dpi=150)
print('saved heatmap')

# --- fig2: Δsep(native) vs act_S1 散点 ---
fig,ax=plt.subplots(figsize=(8.5,6))
colors={'WT':'#1f77b4','TrackF_r1':'#d62728'}
for bg in ['WT','TrackF_r1']:
    sub=piv[piv.bg==bg]
    nb=sub[(sub.F88==native[bg])&(sub.M255=='M')].iloc[0]
    xs=[];ys=[];labels=[]
    for _,r in sub.iterrows():
        if r.F88==native[bg] and r.M255=='M': continue
        xs.append(r.act_S1); ys.append(r.sep-nb.sep)
        labels.append(f'{r.F88}{r.M255}')
    ax.scatter(xs,ys,color=colors[bg],s=55,label=f'{bg} combos',zorder=3,alpha=0.85)
    for (x,y,l) in zip(xs,ys,labels):
        ax.annotate(l,(x,y),fontsize=7,color=colors[bg],alpha=0.9,
                    textcoords='offset points',xytext=(4,-3))
    ax.axhline(nb.sep-nb.sep,color=colors[bg],ls='--',lw=1,alpha=0.5)
# 天然态标注
ax.axhline(0,color='gray',lw=1.2)
ax.set_xlabel('act_S1 (on-target binding fraction, higher better)',fontsize=11)
ax.set_ylabel('Δsep vs native (sep gain, higher better)',fontsize=11)
ax.set_title('Every F88×M255 combo (non-native): gain-vs-retention tradeoff',fontsize=12)
# 区域标注
ax.axvspan(0.70,1.0,ymin=0,color='gold',alpha=0.08)
ax.axhspan(0.10,0.30,xmin=0,color='green',alpha=0.06)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('nom2_tradeoff.png',dpi=150)
print('saved tradeoff')
