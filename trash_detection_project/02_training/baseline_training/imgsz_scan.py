"""학습된 모델을 여러 입력 크기(imgsz)로 평가해 해상도 효과를 잰다. 학습은 하지 않는다.

    uv run python 02_training/baseline_training/imgsz_scan.py                    # 기본: 세 모델 x 640/800/960, val 1,500장
    uv run python 02_training/baseline_training/imgsz_scan.py --n 500 --sizes 640,800

목적
  부록 D 에서 plastic(작은 물체 34.4%)·vinyl(21.2%) 의 성능 저하가 물체 크기 때문이라고
  추정했다. 입력 크기를 키우면 실제로 이 클래스들이 좋아지는지 학습 없이 확인한다.

주의
  - 가중치(best.pt)는 읽기만 하며 바뀌지 않는다.
  - test 는 이미 모델 선택에 썼으므로 **val** 로 측정한다.
  - 우리 이미지는 이미 640x640 으로 저장돼 있다. 800/960 은 확대일 뿐 새 정보가 아니다.
    그래도 물체가 격자(stride) 대비 커져 작은 물체 검출이 개선될 수 있다.
  - 해상도마다 같은 이미지 목록을 쓰므로 비교는 공정하다.

결과: best_train/runs/imgsz_scan/imgsz_scan.txt (+ .csv, 모델별 .json)
"""
# 현재 작업 디렉토리와 상관없이 공용 작업공간 경로를 찾는다.
from pathlib import Path as _WorkspacePath
import sys as _workspace_sys
_workspace_root = next(p for p in _WorkspacePath(__file__).resolve().parents if (p / 'workspace_paths.py').is_file())
_workspace_sys.path.insert(0, str(_workspace_root))
from workspace_paths import workspace_path


import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import yaml

BASE = workspace_path('')
RUNS = workspace_path('', "best_train", "runs")
NOBG = workspace_path('', "best_train", "data_nobg")
BBOX = workspace_path('', "YOLO_7CLASS_10000_PER_CLASS_20260909")
SEG = workspace_path('', "YOLO_SEG_7CLASS_10000_PER_CLASS_20260910_yolo_seg")

MODELS = {
    # 배경 제외 학습(yolo26s) — 지금 가장 정확한 모델. val 도 배경 제외 목록을 쓴다.
    "bbox_yolo26s_nobg": dict(weights=workspace_path('best_train/runs', "bbox_yolo26s_nobg", "weights", "best.pt"),
                              task="detect", val=workspace_path('best_train/data_nobg', "bbox_val.txt"), names=workspace_path('YOLO_7CLASS_10000_PER_CLASS_20260909', "data.yaml")),
    # 배경 포함 학습(yolo11n) — 비교용
    "bbox_yolo11n": dict(weights=workspace_path('best_train/runs', "bbox_yolo11n", "weights", "best.pt"),
                         task="detect", val=workspace_path('YOLO_7CLASS_10000_PER_CLASS_20260909', "images", "val"), names=workspace_path('YOLO_7CLASS_10000_PER_CLASS_20260909', "data.yaml")),
    # 세그(yolo11n-seg) — 마스크가 해상도에 더 민감할 수 있다
    "seg_yolo11n": dict(weights=workspace_path('best_train/runs', "seg_yolo11n", "weights", "best.pt"),
                        task="segment", val=workspace_path('YOLO_SEG_7CLASS_10000_PER_CLASS_20260910_yolo_seg', "images", "val"), names=workspace_path('YOLO_SEG_7CLASS_10000_PER_CLASS_20260910_yolo_seg', "data.yaml")),
}
THIN = ("plastic", "glass", "vinyl")   # 부록 D 에서 지목한 얇고 작은 클래스


def posix(p):
    return str(p).replace("\\", "/")


def val_images(src):
    """val 경로가 목록 파일이면 읽고, 폴더면 파일을 나열한다."""
    src = Path(src)
    if src.suffix == ".txt":
        return [l.strip() for l in open(src, encoding="utf-8") if l.strip()]
    return sorted(posix(src / f) for f in os.listdir(src)
                  if f.lower().endswith((".jpg", ".jpeg", ".png")))


def make_yaml(key, cfg, out, n):
    files = val_images(cfg["val"])
    if n and n < len(files):
        # 전체에서 고르게 뽑는다(앞부분에 배경 사진이 몰려 있어 앞에서 자르면 편향된다)
        files = files[:: max(1, len(files) // n)][:n]
    lst = out / f"{key}_val.txt"
    lst.write_text("\n".join(files) + "\n", encoding="utf-8")
    names = yaml.safe_load(open(cfg["names"], encoding="utf-8"))["names"]
    y = out / f"{key}_val.yaml"
    yaml.safe_dump({"path": posix(out), "train": posix(lst), "val": posix(lst), "test": posix(lst),
                    "nc": len(names), "names": names},
                   open(y, "w", encoding="utf-8"), allow_unicode=True, sort_keys=False)
    return y, len(files)


def evaluate(cfg, yml, imgsz, a, name):
    from ultralytics import YOLO
    r = YOLO(str(cfg["weights"]), task=cfg["task"]).val(
        data=str(yml), split="val", imgsz=imgsz, batch=a.batch, device=a.device,
        rect=False, plots=False, project=str(a.out / "val"), name=name, exist_ok=True, verbose=False)

    def pack(m):
        per = {r.names[int(c)]: float(m.ap[i]) for i, c in enumerate(m.ap_class_index)}
        return dict(map=float(m.map), map50=float(m.map50), p=float(m.mp), r=float(m.mr), per=per)

    out = {"box": pack(r.box), "speed_ms": float(r.speed["inference"])}
    if cfg["task"] == "segment":
        out["mask"] = pack(r.seg)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="640,800,960")
    ap.add_argument("--n", type=int, default=1500, help="val 에서 쓸 이미지 수 (0=전체)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--out", default=str(workspace_path('best_train/runs', "imgsz_scan")))
    ap.add_argument("--force", action="store_true", help="학습이 돌고 있어도 실행")
    a = ap.parse_args()
    a.device = 0 if a.device == "0" else a.device
    a.out = Path(a.out); a.out.mkdir(parents=True, exist_ok=True)
    sizes = [int(s) for s in a.sizes.split(",") if s.strip()]
    keys = [k.strip() for k in a.models.split(",") if k.strip()]

    import torch
    if a.device == 0:
        if not torch.cuda.is_available():
            sys.exit("GPU 를 쓸 수 없습니다.")
        import psutil
        busy = [p.pid for p in psutil.process_iter(["cmdline"])
                if p.info["cmdline"] and any("train.py" in c for c in p.info["cmdline"])]
        if busy and not a.force:
            sys.exit(f"학습 프로세스가 돌고 있습니다 (PID {busy}). 끝난 뒤 실행하세요.")

    L, rows = [], []
    w = L.append
    w("=" * 92)
    w(" 입력 크기(imgsz)별 성능 — 학습 없이 평가만 (val 분할)")
    w("=" * 92)
    w(f"평가 시각 : {datetime.now():%Y-%m-%d %H:%M:%S}   device: {a.device}   batch: {a.batch}")
    w("주의: 원본 이미지가 640x640 이므로 800/960 은 확대입니다. 새 정보가 추가되지는 않습니다.")
    w("")

    for key in keys:
        cfg = MODELS[key]
        if not Path(cfg["weights"]).exists():
            w(f"[{key}] 가중치 없음 - 건너뜀"); continue
        yml, n = make_yaml(key, cfg, a.out, a.n)
        res = {}
        for s in sizes:
            print(f"[{key}] imgsz={s} 평가 중...", flush=True)
            try:
                res[s] = evaluate(cfg, yml, s, a, f"{key}_{s}")
            except Exception as e:
                res[s] = {"error": f"{type(e).__name__}: {e}"}
                print(f"  실패: {res[s]['error']}", flush=True)
        json.dump(res, open(a.out / f"{key}.json", "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1, default=str)

        seg = cfg["task"] == "segment"
        w("#" * 92)
        w(f" [{key}]  val {n:,}장   ({'박스+마스크' if seg else '박스'})")
        w("#" * 92)
        base = res.get(sizes[0], {}).get("box", {}).get("map")
        for kind in (["box", "mask"] if seg else ["box"]):
            kname = "박스" if kind == "box" else "마스크"
            w(f"\n■ 전체 성능 ({kname})")
            w(f"  {'imgsz':>6}{'mAP50-95':>10}{'mAP50':>9}{'P':>8}{'R':>8}{'640대비':>9}{'추론ms':>8}")
            b = res.get(sizes[0], {}).get(kind, {}).get("map")
            for s in sizes:
                v = res[s]
                if "error" in v:
                    w(f"  {s:>6}  실패: {v['error'][:60]}"); continue
                m = v[kind]
                d = f"{m['map'] - b:+.4f}" if b is not None else "-"
                w(f"  {s:>6}{m['map']:>10.4f}{m['map50']:>9.4f}{m['p']:>8.4f}{m['r']:>8.4f}{d:>9}{v['speed_ms']:>8.2f}")
                rows.append([key, kind, s, m["map"], m["map50"], m["p"], m["r"], v["speed_ms"]]
                            + [m["per"].get(c) for c in THIN])
            w(f"\n■ 얇고 작은 클래스 ({kname}, mAP50-95)")
            w(f"  {'imgsz':>6}" + "".join(f"{c:>10}" for c in THIN) + f"{'그 외 평균':>12}")
            for s in sizes:
                v = res[s]
                if "error" in v: continue
                per = v[kind]["per"]
                others = [x for c, x in per.items() if c not in THIN]
                w(f"  {s:>6}" + "".join(f"{per.get(c, float('nan')):>10.4f}" for c in THIN)
                  + f"{sum(others)/max(len(others),1):>12.4f}")
        w("")

    w("=" * 92)
    w(" 읽는 법")
    w("=" * 92)
    w("- 가중치는 바뀌지 않았습니다. 같은 모델을 입력 크기만 달리해 평가한 값입니다.")
    w("- 640 은 학습에 쓴 크기입니다. 800/960 에서 오르면 '추론만 크게' 해서 이득을 볼 수 있습니다.")
    w("- 반대로 떨어지면, 학습 크기와 추론 크기를 맞추는 것이 낫다는 뜻입니다.")
    w("  그 경우 해상도를 키우려면 imgsz=800 으로 다시 학습해야 합니다(추론만 바꿔서는 이득 없음).")
    w("- 추론ms 는 이미지 1장 계산 시간입니다. 크기를 키우면 그만큼 느려집니다.")
    w("- 원본이 640x640 이라 확대의 한계가 있습니다. 원본 고해상도 사진이 남아 있다면")
    w("  데이터셋을 더 큰 크기로 다시 만드는 것이 근본적인 방법입니다.")

    txt = "\n".join(L) + "\n"
    (a.out / "imgsz_scan.txt").write_text(txt, encoding="utf-8")
    with open(a.out / "imgsz_scan.csv", "w", newline="", encoding="utf-8-sig") as f:
        cw = csv.writer(f)
        cw.writerow(["model", "metric", "imgsz", "mAP50-95", "mAP50", "P", "R", "inference_ms",
                     *(f"{c}_mAP50-95" for c in THIN)])
        cw.writerows(rows)
    print(txt)
    print(f"저장: {a.out / 'imgsz_scan.txt'}")


if __name__ == "__main__":
    main()
