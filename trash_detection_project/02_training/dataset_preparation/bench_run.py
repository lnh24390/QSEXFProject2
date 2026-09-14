"""단일 설정 벤치마크 실행기. 결과를 bench_results.jsonl 에 한 줄 추가."""
# 현재 작업 디렉토리와 상관없이 공용 작업공간 경로를 찾는다.
from pathlib import Path as _WorkspacePath
import sys as _workspace_sys
_workspace_root = next(p for p in _WorkspacePath(__file__).resolve().parents if (p / 'workspace_paths.py').is_file())
_workspace_sys.path.insert(0, str(_workspace_root))
from workspace_paths import workspace_path

import sys, json, time, os, argparse, csv

def main():
    os.chdir(str(workspace_path('work')))
    import torch
    from ultralytics import YOLO

    p = argparse.ArgumentParser()
    p.add_argument("--model"); p.add_argument("--batch", type=int); p.add_argument("--workers", type=int)
    p.add_argument("--cache", default="False"); p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--data", default=str(workspace_path('work/bench_1000/data.yaml')))
    p.add_argument("--name"); p.add_argument("--out", default=str(workspace_path('work/bench_results.jsonl')))
    p.add_argument("--device", default="0", help="0 = GPU, cpu = CPU 비교용")
    a = p.parse_args()
    dev = 0 if a.device == "0" else a.device
    cache = {"False": False, "True": True, "ram": "ram", "disk": "disk"}[a.cache]

    if a.device == "0":
        torch.cuda.reset_peak_memory_stats()
    rec = {"name": a.name, "model": a.model, "batch": a.batch, "workers": a.workers,
           "cache": a.cache, "epochs": a.epochs, "device": a.device,
           "data": os.path.basename(os.path.dirname(a.data))}
    t0 = time.time()
    try:
        model_path = _WorkspacePath(a.model)
        y = YOLO(str(model_path if model_path.is_absolute() else workspace_path('work', a.model)))
        r = y.train(data=a.data, epochs=a.epochs, imgsz=640, batch=a.batch, workers=a.workers,
                    cache=cache, device=dev, amp=True, pretrained=True, val=True, plots=False,
                    project=str(workspace_path('work/runs')), name=a.name, exist_ok=True, seed=0, deterministic=False,
                    patience=0, verbose=False)
        rec["wall_s"] = round(time.time() - t0, 1)
        rows = list(csv.DictReader(open(os.path.join(str(r.save_dir), "results.csv"))))
        key = next(k for k in rows[0] if k.strip() == "time")
        cum = [float(x[key]) for x in rows]
        per = [cum[0]] + [cum[i] - cum[i-1] for i in range(1, len(cum))]
        rec["epoch_times"] = [round(v, 2) for v in per]
        steady = per[1:] if len(per) > 1 else per
        rec["epoch_mean_s"] = round(sum(steady) / len(steady), 2)
        rec["first_epoch_s"] = round(per[0], 2)
        ntrain = len(os.listdir(os.path.join(os.path.dirname(a.data), "images", "train")))
        rec["train_images"] = ntrain
        rec["img_per_s"] = round(ntrain / rec["epoch_mean_s"], 1)
        rec["gpu_peak_gb"] = round(torch.cuda.max_memory_reserved() / 1e9, 2) if dev == 0 else None
        m = getattr(r, "results_dict", {}) or {}
        rec["map50_95"] = round(float(m.get("metrics/mAP50-95(B)", 0)), 4)
        rec["map50"] = round(float(m.get("metrics/mAP50(B)", 0)), 4)
        try:
            names = y.model.names
            rec["class_recall"] = {names[int(c)]: round(float(v), 4)
                                   for c, v in zip(r.box.ap_class_index, r.box.r)}
        except Exception as e:
            rec["class_recall"] = {}; rec["recall_note"] = f"{type(e).__name__}: {e}"
        rec["status"] = "ok"
    except torch.cuda.OutOfMemoryError as e:
        rec["status"] = "OOM"; rec["error"] = str(e)[:200]
    except Exception as e:
        rec["status"] = "FAIL"; rec["error"] = f"{type(e).__name__}: {str(e)[:400]}"
    with open(a.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print("RESULT " + json.dumps({k: rec.get(k) for k in
          ("name","status","epoch_mean_s","img_per_s","gpu_peak_gb","map50_95","error")},
          ensure_ascii=False))

if __name__ == "__main__":
    main()
