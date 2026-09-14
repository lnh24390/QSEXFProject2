# 현재 작업 디렉토리와 상관없이 공용 작업공간 경로를 찾는다.
from pathlib import Path as _WorkspacePath
import sys as _workspace_sys
_workspace_root = next(p for p in _WorkspacePath(__file__).resolve().parents if (p / 'workspace_paths.py').is_file())
_workspace_sys.path.insert(0, str(_workspace_root))
from workspace_paths import workspace_path

import csv, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
for c in ("Malgun Gothic","MalgunGothic","Gulim"):
    if any(c.lower()==f.name.lower() for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"]=c; break
plt.rcParams["axes.unicode_minus"]=False
OUT=str(workspace_path('work/figs'))

rows=[r for r in csv.reader(open(str(workspace_path('work/gpu_log.csv')))) if len(r)>=6]
util=[float(r[1]) for r in rows]; mem=[float(r[3])/1024 for r in rows]
t=list(range(len(util)))
fig,ax=plt.subplots(figsize=(10,4.2))
ax.fill_between(t,util,color="#3b82f6",alpha=.35); ax.plot(t,util,color="#1d4ed8",lw=1.4,label="GPU 사용률 (%)")
ax2=ax.twinx(); ax2.plot(t,mem,color="#ea580c",lw=1.6,ls="--",label="GPU 메모리 (GB)")
ax2.set_ylabel("GPU 메모리 (GB)",color="#ea580c"); ax2.set_ylim(0,8.19); ax2.tick_params(axis="y",colors="#ea580c")
tr=[i for i,m in enumerate(mem) if m>=2.0]
ax.axvspan(min(tr),max(tr),color="#22c55e",alpha=.10)
ax.text((min(tr)+max(tr))/2,103,"학습 프로세스 GPU 점유 구간 (58초)",ha="center",fontsize=9,color="#15803d")
ax.axhline(71.5,ls=":",c="#1d4ed8",lw=1); ax.text(1,73,"중앙값 71.5%",fontsize=8,color="#1d4ed8")
ax.set_xlabel("경과 시간 (초)"); ax.set_ylabel("GPU 사용률 (%)",color="#1d4ed8"); ax.set_ylim(0,110)
ax.tick_params(axis="y",colors="#1d4ed8"); ax.grid(alpha=.3)
ax.set_title("그림 7. 학습 중 GPU 실사용 검증 — YOLO11n, batch=16, 1초 간격 85회 샘플링\n사용률 최대 92%, 메모리 최대 3.65GB → CUDA 로 실제 연산 중임을 확인")
fig.tight_layout(); fig.savefig(str(workspace_path('work/figs', f'fig7_gpu_util.png')),dpi=140); plt.close(fig)

fig,ax=plt.subplots(1,2,figsize=(10,4.2))
lbl=["CPU\n(i7-13700, 16스레드)","CUDA\n(RTX 4060)"]; val=[550.70,21.36]
b=ax[0].bar(lbl,val,color=["#94a3b8","#2563eb"])
for r,v in zip(b,val): ax[0].text(r.get_x()+r.get_width()/2,v+15,f"{v:.1f}초",ha="center",fontsize=10,fontweight="bold")
ax[0].set_ylabel("에포크 평균 시간 (초)"); ax[0].set_ylim(0,640); ax[0].grid(alpha=.3,axis="y")
ax[0].set_title("에포크 시간 — CUDA 가 25.8배 빠름")
val2=[6.1,157.1]
b=ax[1].bar(lbl,val2,color=["#94a3b8","#2563eb"])
for r,v in zip(b,val2): ax[1].text(r.get_x()+r.get_width()/2,v+4,f"{v:.1f}",ha="center",fontsize=10,fontweight="bold")
ax[1].set_ylabel("초당 학습 이미지 (img/s)"); ax[1].set_ylim(0,185); ax[1].grid(alpha=.3,axis="y")
ax[1].set_title("처리량")
fig.suptitle("그림 8. CPU 전용 torch vs CUDA torch — 동일 설정(YOLO11n, batch=16, workers=4, 소량 세트)",fontsize=11)
fig.tight_layout(); fig.savefig(str(workspace_path('work/figs', f'fig8_cpu_vs_cuda.png')),dpi=140); plt.close(fig)
print("saved fig7, fig8")
