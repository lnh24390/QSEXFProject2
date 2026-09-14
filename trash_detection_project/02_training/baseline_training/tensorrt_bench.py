"""학습한 모델을 TensorRT 엔진으로 바꾸고, 변환 전후의 예측 속도와 정확도를 비교한다.

    uv run python 02_training/baseline_training/tensorrt_bench.py bbox seg              # 배경 포함 yolo11n 두 모델
    uv run python 02_training/baseline_training/tensorrt_bench.py bbox_nobg seg_nobg    # 배경 제외 yolo26s 두 모델

TensorRT 는 NVIDIA GPU 전용 추론 엔진이다. 학습을 빠르게 하는 것이 아니라,
학습이 끝난 모델의 '예측'을 빠르게 한다. 모델을 이 PC 의 GPU 에 맞춰 최적화한
.engine 파일로 바꾸고, FP16(반정밀도)으로 계산해 속도를 올린다.

비교하는 세 가지 (같은 test 분할, 이미지 1장씩 = 실제 서비스와 같은 조건)
    PyTorch FP32   학습 결과 그대로 (best.pt)
    PyTorch FP16   best.pt 를 반정밀도로 실행 — 속도 향상 중 FP16 몫을 가려내기 위함
    TensorRT FP16  best.engine

결과: best_train/runs/tensorrt/tensorrt_summary.txt (모델별 결과는 <모델>.json)
엔진 파일: best_train/runs/<학습 폴더>/weights/best.engine

주의: .engine 은 만든 PC 의 GPU 종류와 TensorRT 버전에 묶인다. 다른 PC 에서는 다시 만들어야 한다.
"""
# 현재 작업 디렉토리와 상관없이 공용 작업공간 경로를 찾는다.
from pathlib import Path as _WorkspacePath
import sys as _workspace_sys
_workspace_root = next(p for p in _WorkspacePath(__file__).resolve().parents if (p / 'workspace_paths.py').is_file())
_workspace_sys.path.insert(0, str(_workspace_root))
from workspace_paths import workspace_path


import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

BASE = workspace_path('')
RUNS = workspace_path('', "best_train", "runs")
NOBG = workspace_path('', "best_train", "data_nobg")

MODELS = {
    "bbox": dict(run="bbox_yolo11n", task="detect",
                 data=workspace_path('', "YOLO_7CLASS_10000_PER_CLASS_20260909", "data.yaml")),
    "seg": dict(run="seg_yolo11n", task="segment",
                data=workspace_path('', "YOLO_SEG_7CLASS_10000_PER_CLASS_20260910_yolo_seg", "data.yaml")),
    "bbox_nobg": dict(run="bbox_yolo26s_nobg", task="detect", data=workspace_path('best_train/data_nobg', "bbox_nobg.yaml")),
    "seg_nobg": dict(run="seg_yolo26s_nobg", task="segment", data=workspace_path('best_train/data_nobg', "seg_nobg.yaml")),
}
FORMATS = {"pt32": "PyTorch FP32", "pt16": "PyTorch FP16", "engine": "TensorRT FP16"}


def posix(p):
    return str(p).replace("\\", "/")


def test_images(data_yaml):
    """data.yaml 의 test 가 폴더든 목록 파일이든 이미지 경로 목록으로 바꾼다."""
    d = yaml.safe_load(open(data_yaml, encoding="utf-8"))
    t = Path(d["test"])
    if not t.is_absolute():
        t = Path(d.get("path", Path(data_yaml).parent)) / t
    if t.suffix == ".txt":
        files = [l.strip() for l in open(t, encoding="utf-8") if l.strip()]
    else:
        files = sorted(posix(p) for p in t.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    return files, d["names"]


def make_yaml(key, data_yaml, out, limit):
    files, names = test_images(data_yaml)
    if limit:
        # 이름순 앞부분은 bg_coco_* 배경 사진이 몰려 있어 전체에서 고르게 뽑는다
        files = files[::max(1, len(files) // limit)][:limit]
    lst = out / f"{key}_test.txt"
    lst.write_text("\n".join(files) + "\n", encoding="utf-8")
    y = out / f"{key}_test.yaml"
    yaml.safe_dump({"path": posix(out), "train": posix(lst), "val": posix(lst), "test": posix(lst),
                    "nc": len(names), "names": names},
                   open(y, "w", encoding="utf-8"), allow_unicode=True, sort_keys=False)
    return y, len(files)


def evaluate(weights, task, yml, a, name, quantize):
    from ultralytics import YOLO
    kw = dict(data=str(yml), split="test", imgsz=640, batch=1, device=a.device, rect=False,
              plots=False, project=str(a.out / "val"), name=name, exist_ok=True, verbose=False)
    if quantize:
        kw["quantize"] = quantize
    t0 = time.time()
    r = YOLO(str(weights), task=task).val(**kw)
    s = r.speed
    res = dict(pre=s["preprocess"], inf=s["inference"], post=s["postprocess"],
               wall=time.time() - t0, box=float(r.box.map), box50=float(r.box.map50))
    if task == "segment":
        res["mask"], res["mask50"] = float(r.seg.map), float(r.seg.map50)
    return res


def export_engine(pt, a):
    eng = pt.with_suffix(".engine")
    if eng.exists() and eng.stat().st_mtime > pt.stat().st_mtime and not a.rebuild:
        print(f"기존 엔진 사용: {eng}")
        return eng, None
    from ultralytics import YOLO
    t0 = time.time()
    out = YOLO(str(pt)).export(format="engine", quantize=16, imgsz=640, batch=1,
                               device=a.device, workspace=a.workspace, simplify=True)
    return Path(out), time.time() - t0


def mb(p):
    return Path(p).stat().st_size / 1e6 if Path(p).exists() else None


def write_summary(out):
    L = []
    w = L.append
    w("=" * 96)
    w(" TensorRT 가속 비교 (test 분할, 이미지 1장씩 예측)")
    w("=" * 96)
    for f in sorted(out.glob("*.json")):
        R = json.load(open(f, encoding="utf-8"))
        w("")
        w(f"■ {R['key']}  ({R['run']}, {R['task']})   평가 {R['time']}")
        w(f"  test {R['n_test']:,}장  |  GPU {R['gpu']}  |  torch {R['torch']}  |  TensorRT {R['trt']}")
        if R.get("export_s") is not None:
            w(f"  엔진 변환 시간 {R['export_s']/60:.1f}분  |  엔진 파일 {R['engine']}")
        seg = R["task"] == "segment"
        w(f"  {'형식':<16}{'파일MB':>8}{'box mAP50-95':>14}" + (f"{'mask mAP50-95':>15}" if seg else "")
          + f"{'전처리ms':>10}{'추론ms':>9}{'후처리ms':>10}{'합계ms':>9}{'FPS':>8}{'추론 배수':>10}")
        base = R["rows"].get("pt32", {}).get("inf")
        for k, label in FORMATS.items():
            v = R["rows"].get(k)
            if not v:
                continue
            if "error" in v:
                w(f"  {label:<16}  실패: {v['error'][:70]}")
                continue
            tot = v["pre"] + v["inf"] + v["post"]
            size = f"{v['size']:8.1f}" if v.get("size") else f"{'-':>8}"
            w(f"  {label:<16}{size}{v['box']:14.4f}" + (f"{v['mask']:15.4f}" if seg else "")
              + f"{v['pre']:10.2f}{v['inf']:9.2f}{v['post']:10.2f}{tot:9.2f}{1000/tot:8.1f}"
              + (f"{base / v['inf']:9.2f}x" if base and v["inf"] > 0 else f"{'-':>10}"))
        e, p = R["rows"].get("engine", {}), R["rows"].get("pt32", {})
        if "box" in e and "box" in p:
            d = e["box"] - p["box"]
            w(f"  정확도 변화(TensorRT - PyTorch FP32): box mAP50-95 {d:+.4f}"
              + (f", mask mAP50-95 {e['mask'] - p['mask']:+.4f}" if seg else "")
              + ("   <- 0.01 이상 차이, 확인 필요" if abs(d) >= 0.01 else "   (차이 작음)"))
    w("")
    w("-" * 96)
    w("- 추론ms 는 GPU 가 모델을 계산한 시간, 합계ms 는 전처리+추론+후처리입니다. 이미지 파일 읽기는 포함하지 않습니다.")
    w("- 추론 배수 = PyTorch FP32 추론ms / 해당 형식 추론ms. 후처리(NMS, 마스크 계산)는 TensorRT 로 빨라지지 않습니다.")
    w("- FPS 는 합계ms 기준 초당 처리 장수입니다. 카메라 영상 등 실제 입력에서는 읽기·표시 시간이 더해집니다.")
    w("- .engine 은 이 PC(RTX 4060)와 TensorRT 버전 전용입니다. 다른 PC 에서는 다시 변환하세요.")
    (out / "tensorrt_summary.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("keys", nargs="+", choices=list(MODELS))
    ap.add_argument("--device", default="0")
    ap.add_argument("--formats", default="pt32,pt16,engine")
    ap.add_argument("--limit", type=int, default=0, help="동작 확인용: test 에서 N장만 사용")
    ap.add_argument("--workspace", type=float, default=4, help="TensorRT 변환 작업 메모리(GiB)")
    ap.add_argument("--rebuild", action="store_true", help="엔진이 있어도 다시 변환")
    ap.add_argument("--weights", default="", help="동작 확인용: 가중치 경로를 직접 지정")
    ap.add_argument("--out", default=str(workspace_path('best_train/runs', "tensorrt")))
    ap.add_argument("--force", action="store_true", help="학습이 돌고 있어도 GPU 로 실행")
    a = ap.parse_args()
    a.device = 0 if a.device == "0" else a.device
    a.out = Path(a.out)
    a.out.mkdir(parents=True, exist_ok=True)
    formats = [f.strip() for f in a.formats.split(",") if f.strip()]

    import torch
    trt_ver = "-"
    try:
        import tensorrt
        trt_ver = tensorrt.__version__
    except Exception:
        pass
    if a.device == 0:
        if not torch.cuda.is_available():
            sys.exit("GPU 를 쓸 수 없습니다. torch CUDA 빌드를 확인하세요.")
        import psutil
        busy = [p.pid for p in psutil.process_iter(["cmdline"])
                if p.info["cmdline"] and any("train.py" in c for c in p.info["cmdline"])]
        if busy and not a.force:
            sys.exit(f"학습 프로세스가 돌고 있습니다 (PID {busy}). 끝난 뒤 실행하거나 --force 를 쓰세요.")
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() and a.device == 0 else str(a.device)

    for key in a.keys:
        cfg = MODELS[key]
        pt = Path(a.weights) if a.weights else workspace_path('best_train/runs', cfg["run"], "weights", "best.pt")
        if not pt.exists():
            print(f"[{key}] 가중치 없음, 건너뜀: {pt}")
            continue
        yml, n = make_yaml(key, cfg["data"], a.out, a.limit)
        R = dict(key=key, run=cfg["run"], task=cfg["task"], n_test=n, gpu=gpu,
                 torch=torch.__version__, trt=trt_ver, time=f"{datetime.now():%Y-%m-%d %H:%M}",
                 export_s=None, engine=None, rows={})
        for fmt in formats:
            print(f"[{key}] {FORMATS[fmt]} 평가 중...", flush=True)
            try:
                if fmt == "engine":
                    eng, R["export_s"] = export_engine(pt, a)
                    R["engine"] = str(eng)
                    row = evaluate(eng, cfg["task"], yml, a, f"{key}_{fmt}", None)
                    row["size"] = mb(eng)
                else:
                    row = evaluate(pt, cfg["task"], yml, a, f"{key}_{fmt}", 16 if fmt == "pt16" else 32)
                    row["size"] = mb(pt)
            except Exception as e:  # 한 형식이 실패해도 나머지 결과는 남긴다
                row = dict(error=f"{type(e).__name__}: {e}")
                print(f"[{key}] {FORMATS[fmt]} 실패: {row['error']}", flush=True)
            R["rows"][fmt] = row
            json.dump(R, open(a.out / f"{key}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    print(write_summary(a.out))
    print(f"저장: {a.out / 'tensorrt_summary.txt'}")


if __name__ == "__main__":
    main()
