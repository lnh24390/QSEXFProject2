# 현재 작업 디렉토리와 상관없이 공용 작업공간 경로를 찾는다.
from pathlib import Path as _WorkspacePath
import sys as _workspace_sys
_workspace_root = next(p for p in _WorkspacePath(__file__).resolve().parents if (p / 'workspace_paths.py').is_file())
_workspace_sys.path.insert(0, str(_workspace_root))
from workspace_paths import workspace_path

import json, os, collections
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
for cand in ("Malgun Gothic","MalgunGothic","Gulim"):
    if any(cand.lower()==f.name.lower() for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"]=cand; break
plt.rcParams["axes.unicode_minus"]=False
OUT=str(workspace_path('work/figs')); os.makedirs(OUT,exist_ok=True)
R=[json.loads(l) for l in open(str(workspace_path('work/bench_results.jsonl')),encoding="utf-8")]
R=[r for r in R if r["status"]=="ok"]
def sel(model,workers=4,cache="False"):
    return sorted([r for r in R if r["model"]==model and r["workers"]==workers and r["cache"]==cache],
                  key=lambda r:r["batch"])
y11,y26=sel("yolo11n.pt"),sel("yolo26n.pt")
C11,C26="#2563eb","#ea580c"

# 그림1: batch vs 에포크 시간
fig,ax=plt.subplots(1,2,figsize=(11,4.2))
for d,c,lb in ((y11,C11,"YOLO11n"),(y26,C26,"YOLO26n")):
    b=[r["batch"] for r in d]
    ax[0].plot(b,[r["epoch_mean_s"] for r in d],"o-",color=c,label=lb,lw=2)
    ax[1].plot(b,[r["img_per_s"] for r in d],"o-",color=c,label=lb,lw=2)
    for r in d:
        ax[0].annotate(f"{r['epoch_mean_s']:.1f}",(r["batch"],r["epoch_mean_s"]),
                       textcoords="offset points",xytext=(0,8),ha="center",fontsize=8,color=c)
ax[0].set_xlabel("batch"); ax[0].set_ylabel("에포크 평균 시간 (초)"); ax[0].set_title("batch 별 에포크 시간 (낮을수록 좋음)")
ax[1].set_xlabel("batch"); ax[1].set_ylabel("초당 학습 이미지 (img/s)"); ax[1].set_title("batch 별 처리량 (높을수록 좋음)")
for a in ax: a.grid(alpha=.3); a.legend(); a.set_xticks([8,16,24,32,48])
fig.suptitle("그림 1. batch 크기 효과 — workers=4, 소량 세트 3,356장, RTX 4060",fontsize=11)
fig.tight_layout(); fig.savefig(str(workspace_path('work/figs', f'fig1_batch.png')),dpi=140); plt.close(fig)

# 그림2: batch vs GPU 메모리
fig,ax=plt.subplots(figsize=(7,4.2))
for d,c,lb in ((y11,C11,"YOLO11n"),(y26,C26,"YOLO26n")):
    ax.plot([r["batch"] for r in d],[r["gpu_peak_gb"] for r in d],"o-",color=c,label=lb,lw=2)
    for r in d: ax.annotate(f"{r['gpu_peak_gb']:.2f}",(r["batch"],r["gpu_peak_gb"]),
                   textcoords="offset points",xytext=(0,7),ha="center",fontsize=8,color=c)
ax.axhline(8.19,ls="--",c="#dc2626"); ax.text(9,8.25,"RTX 4060 전체 8.19GB",color="#dc2626",fontsize=9)
ax.axhline(6.2,ls=":",c="#6b7280"); ax.text(9,6.3,"데스크톱 약 2GB 사용 시 실사용 한계",color="#6b7280",fontsize=9)
ax.set_xlabel("batch"); ax.set_ylabel("최대 GPU 메모리 (GB)"); ax.set_xticks([8,16,24,32,48])
ax.set_ylim(0,9); ax.grid(alpha=.3); ax.legend(loc="upper left")
ax.set_title("그림 2. batch 별 최대 GPU 메모리")
fig.tight_layout(); fig.savefig(str(workspace_path('work/figs', f'fig2_memory.png')),dpi=140); plt.close(fig)

# 그림3: workers 효과
fig,ax=plt.subplots(figsize=(7,4.2))
w11=sorted([r for r in R if r["model"]=="yolo11n.pt" and r["batch"]==24 and r["cache"]=="False"],key=lambda r:r["workers"])
lbl=[str(r["workers"]) for r in w11]; val=[r["epoch_mean_s"] for r in w11]
bars=ax.bar(lbl,val,color=["#94a3b8" if r["workers"]==0 else C11 for r in w11])
for b,v in zip(bars,val): ax.text(b.get_x()+b.get_width()/2,v+1,f"{v:.2f}s",ha="center",fontsize=9)
ax.set_xlabel("workers"); ax.set_ylabel("에포크 평균 시간 (초)")
ax.set_title("그림 3. workers 효과 — YOLO11n, batch=24\n4→8→12 는 차이 없음(≤1.4%). 0 은 2.6배 느림 = 워커 자체는 필수")
ax.grid(alpha=.3,axis="y"); fig.tight_layout(); fig.savefig(str(workspace_path('work/figs', f'fig3_workers.png')),dpi=140); plt.close(fig)

# 그림4: 속도 vs 정확도
fig,ax=plt.subplots(figsize=(7.5,4.6))
for d,c,lb in ((y11,C11,"YOLO11n"),(y26,C26,"YOLO26n")):
    ax.scatter([r["img_per_s"] for r in d],[r["map50_95"] for r in d],color=c,s=70,label=lb,zorder=3)
    for r in d: ax.annotate(f"b{r['batch']}",(r["img_per_s"],r["map50_95"]),
                  textcoords="offset points",xytext=(6,4),fontsize=8,color=c)
ax.set_xlabel("초당 학습 이미지 (img/s) →  빠름"); ax.set_ylabel("mAP50-95 (4에포크)")
ax.set_title("그림 4. 속도-정확도 관계 — 오른쪽 위가 유리\n(4에포크 값은 참고용 조기 지표이며 최종 성능이 아님)")
ax.grid(alpha=.3); ax.legend(); fig.tight_layout(); fig.savefig(str(workspace_path('work/figs', f'fig4_speed_acc.png')),dpi=140); plt.close(fig)

# 그림5: 분할 조정 전후
before={"paper":(80.1,14.9,5.0),"plastic":(80.1,14.9,5.0),"can":(80.2,14.8,5.0),"glass":(82.8,14.8,2.4),
        "battery":(96.6,3.0,0.4),"vinyl":(80.4,14.7,4.9),"general":(92.3,7.4,0.3)}
after={"paper":(79.6,15.0,5.3),"plastic":(79.8,15.0,5.2),"can":(79.9,15.0,5.1),"glass":(80.0,15.0,5.0),
       "battery":(80.0,15.0,5.0),"vinyl":(80.0,15.0,5.0),"general":(80.0,15.0,5.0)}
fig,ax=plt.subplots(1,2,figsize=(11,4.2),sharey=True)
ks=list(before); cols=["#3b82f6","#f59e0b","#10b981"]
for i,(d,t) in enumerate(((before,"조정 전"),(after,"조정 후"))):
    bot=[0]*len(ks)
    for j,(nm,c) in enumerate(zip(["train","val","test"],cols)):
        v=[d[k][j] for k in ks]
        ax[i].bar(ks,v,bottom=bot,color=c,label=nm if i==0 else None)
        bot=[a+b for a,b in zip(bot,v)]
    ax[i].set_title(t); ax[i].tick_params(axis="x",rotation=45); ax[i].set_ylim(0,100)
    ax[i].axhline(80,ls="--",c="k",lw=.8); ax[i].axhline(95,ls="--",c="k",lw=.8)
ax[0].set_ylabel("분할 비율 (%)"); fig.legend(loc="upper right",ncol=3,fontsize=9)
fig.suptitle("그림 5. 클래스별 분할 비율 조정 전후 — 점선은 목표 80% / 95% 경계",fontsize=11)
fig.tight_layout(); fig.savefig(str(workspace_path('work/figs', f'fig5_split.png')),dpi=140); plt.close(fig)

# 그림6: 조정 전후 val/test 이미지 수
fig,ax=plt.subplots(figsize=(8,4.2))
import numpy as np
x=np.arange(len(ks)); w=0.2
bv=[{"paper":1491,"plastic":1491,"can":1485,"glass":1477,"battery":300,"vinyl":1472,"general":740}[k] for k in ks]
av=[{"paper":1504,"plastic":1503,"can":1500,"glass":1500,"battery":1500,"vinyl":1501,"general":1500}[k] for k in ks]
bt=[{"paper":496,"plastic":499,"can":500,"glass":238,"battery":40,"vinyl":489,"general":26}[k] for k in ks]
at=[{"paper":535,"plastic":518,"can":508,"glass":500,"battery":500,"vinyl":502,"general":500}[k] for k in ks]
ax.bar(x-1.5*w,bv,w,label="val 조정 전",color="#fcd34d"); ax.bar(x-0.5*w,av,w,label="val 조정 후",color="#f59e0b")
ax.bar(x+0.5*w,bt,w,label="test 조정 전",color="#6ee7b7"); ax.bar(x+1.5*w,at,w,label="test 조정 후",color="#10b981")
ax.set_xticks(x); ax.set_xticklabels(ks,rotation=45); ax.set_ylabel("이미지 수")
ax.set_title("그림 6. 평가용 이미지 수 — battery 40→500, general 26→500 (test)")
ax.legend(fontsize=8); ax.grid(alpha=.3,axis="y"); fig.tight_layout()
fig.savefig(str(workspace_path('work/figs', f'fig6_eval_images.png')),dpi=140); plt.close(fig)
print("저장:",os.listdir(OUT))
