"""배경 이미지를 넣고 학습한 모델과 뺀 모델을 같은 test 목록으로 비교한다.

    uv run python 02_training/baseline_training/compare_bg.py                 # 네 모델 모두, GPU
    uv run python 02_training/baseline_training/compare_bg.py --families bbox # 검출만

비교 방법
  - 검출 모델(bbox, bbox_nobg)은 검출 데이터셋 test 로만, 세그 모델(seg, seg_nobg)은
    세그 데이터셋 test 로만 평가한다. 두 데이터셋은 분할이 달라서 섞으면 학습 이미지가
    평가에 들어간다.
  - 같은 test 를 두 가지로 쓴다.
      배경 포함 test : 라벨 이미지 + 배경 150장 (배경에서 오탐하면 점수가 깎인다)
      배경 제외 test : 라벨 이미지만           (물체를 찾는 성능만 본다)
  - 배경 150장만 따로 예측해, 아무것도 없는 사진에서 몇 장·몇 개를 잘못 검출하는지 센다.

결과: best_train/runs/compare_bg/compare_result.txt, compare_result.csv

test 분할은 최종 평가용이다. 이 결과를 보고 설정을 다시 고르면 test 가 오염된다.
"""
# 현재 작업 디렉토리와 상관없이 공용 작업공간 경로를 찾는다.
from pathlib import Path as _WorkspacePath
import sys as _workspace_sys
_workspace_root = next(p for p in _WorkspacePath(__file__).resolve().parents if (p / 'workspace_paths.py').is_file())
_workspace_sys.path.insert(0, str(_workspace_root))
from workspace_paths import workspace_path
_workspace_sys.path.insert(0, str(workspace_path("02_training")))
from model_analysis.evaluation import evaluate_model
from model_analysis.schema import unique_dir


import argparse
import collections
import csv
import os
import sys
from datetime import datetime
from pathlib import Path

import yaml

BASE = workspace_path('')
RUNS = workspace_path('', "best_train", "runs")
NOBG = workspace_path('', "best_train", "data_nobg")

FAMILIES = {
    "bbox": dict(
        title="검출", task="detect",
        root=workspace_path('', "YOLO_7CLASS_10000_PER_CLASS_20260909"),
        nobg_list=workspace_path('best_train/data_nobg', "bbox_test.txt"),
        models={"bbox": workspace_path('best_train/runs', "bbox_yolo11n", "weights", "best.pt"),
                "bbox_nobg": workspace_path('best_train/runs', "bbox_yolo26s_nobg", "weights", "best.pt")},
    ),
    "seg": dict(
        title="세그", task="segment",
        root=workspace_path('', "YOLO_SEG_7CLASS_10000_PER_CLASS_20260910_yolo_seg"),
        nobg_list=workspace_path('best_train/data_nobg', "seg_test.txt"),
        models={"seg": workspace_path('best_train/runs', "seg_yolo11n", "weights", "best.pt"),
                "seg_nobg": workspace_path('best_train/runs', "seg_yolo26s_nobg", "weights", "best.pt")},
    ),
}
# 배경 제외 학습은 요청에 따라 yolo26s 로 했으므로 모델 이름을 함께 표시한다
LABEL = {"bbox": "yolo11n 배경 포함", "bbox_nobg": "yolo26s 배경 제외",
         "seg": "yolo11n-seg 배경 포함", "seg_nobg": "yolo26s-seg 배경 제외"}
CONFS = (0.25, 0.5)


def posix(p):
    return str(p).replace("\\", "/")


def build_sets(fam, cfg, out, limit):
    """test 목록 3개(배경 포함, 배경 제외, 배경만)와 평가용 yaml 을 만든다."""
    test_dir = cfg["root"] / "images" / "test"
    every = sorted(posix(test_dir / f) for f in os.listdir(test_dir))
    bg = [p for p in every if os.path.basename(p).startswith("bg_coco_")]
    labeled = [l.strip() for l in open(cfg["nobg_list"], encoding="utf-8") if l.strip()]
    if limit:
        labeled, bg = labeled[:limit], bg[:max(3, limit // 4)]
    else:
        # 배경 포함 test = 라벨 이미지 + 배경 이 test 폴더 전체와 정확히 같아야 한다
        assert sorted(labeled + bg) == every, f"{fam}: test 목록이 폴더와 다릅니다"

    names = yaml.safe_load(open(cfg["root"] / "data.yaml", encoding="utf-8"))["names"]
    sets = {}
    for key, files in (("with_bg", labeled + bg), ("no_bg", labeled), ("bg_only", bg)):
        lst = out / f"{fam}_{key}_test.txt"
        lst.write_text("\n".join(files) + "\n", encoding="utf-8")
        y = out / f"{fam}_{key}.yaml"
        yaml.safe_dump({"path": posix(out), "train": posix(lst), "val": posix(lst),
                        "test": posix(lst), "nc": len(names), "names": names},
                       open(y, "w", encoding="utf-8"), allow_unicode=True, sort_keys=False)
        sets[key] = dict(yaml=y, list=lst, n=len(files))
    sets["n_labeled"], sets["n_bg"] = len(labeled), len(bg)
    return sets


def summarize(m, names):
    per = {names[int(c)]: dict(p=float(m.p[i]), r=float(m.r[i]),
                               map50=float(m.ap50[i]), map=float(m.ap[i]))
           for i, c in enumerate(m.ap_class_index)}
    return dict(map=float(m.map), map50=float(m.map50), p=float(m.mp), r=float(m.mr), per=per)


def evaluate(weights, yaml_path, task, a, name):
    from ultralytics import YOLO
    r = evaluate_model(weights, yaml_path, split="test", task=task, batch=a.batch, device=a.device,
                       source="background_comparison", name=name)
    res = {"box": summarize(r.box, r.names)}
    if task == "segment":
        res["mask"] = summarize(r.seg, r.names)
    return res


def background_fp(weights, bg_list, a):
    """배경 사진에서 잘못 검출한 이미지 수와 검출 수를 신뢰도 기준별로 센다."""
    from ultralytics import YOLO
    m = YOLO(str(weights))
    names = m.names
    img_hit = {t: 0 for t in CONFS}
    det = {t: 0 for t in CONFS}
    by_cls = collections.Counter()
    n = 0
    for r in m.predict(source=str(bg_list), imgsz=640, conf=min(CONFS), device=a.device,
                       stream=True, verbose=False, batch=a.batch):
        n += 1
        if r.boxes is None or len(r.boxes) == 0:
            continue
        conf = r.boxes.conf.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy()
        for t in CONFS:
            k = int((conf >= t).sum())
            det[t] += k
            img_hit[t] += int(k > 0)
        for c in cls:
            by_cls[names[int(c)]] += 1
    return dict(n=n, img_hit=img_hit, det=det, by_cls=by_cls)


def f4(x):
    return "   -   " if x is None else f"{x:7.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="0")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="동작 확인용: test 에서 N장만 사용")
    ap.add_argument("--out", default=str(workspace_path('best_train/runs', "compare_bg")))
    ap.add_argument("--families", default="bbox,seg")
    ap.add_argument("--models", nargs="*", default=[], help="가중치 바꾸기: 이름=경로")
    ap.add_argument("--force", action="store_true", help="학습이 돌고 있어도 GPU 로 실행")
    a = ap.parse_args()
    a.device = 0 if a.device == "0" else a.device
    a.out = unique_dir(Path(a.out), "background_comparison")
    a.out.mkdir(parents=True, exist_ok=True)
    for kv in a.models:
        k, v = kv.split("=", 1)
        for cfg in FAMILIES.values():
            if k in cfg["models"]:
                cfg["models"][k] = Path(v)

    if a.device == 0:
        import torch
        if not torch.cuda.is_available():
            sys.exit("GPU 를 쓸 수 없습니다. torch CUDA 빌드를 확인하세요.")
        # 학습 중에 GPU 평가를 돌리면 학습이 느려지고 메모리가 모자랄 수 있다
        import psutil
        busy = [p.pid for p in psutil.process_iter(["cmdline"])
                if p.info["cmdline"] and any("train.py" in c for c in p.info["cmdline"])]
        if busy and not a.force:
            sys.exit(f"학습 프로세스가 돌고 있습니다 (PID {busy}). 끝난 뒤 실행하거나 --force 를 쓰세요.")

    L = []
    w = L.append
    rows = []
    w("=" * 78)
    w(" 배경 이미지 포함 / 제외 학습 비교 (test 분할)")
    w("=" * 78)
    w(f"평가 시각 : {datetime.now():%Y-%m-%d %H:%M:%S}")
    w(f"device    : {a.device}" + (f"   (동작 확인용 --limit {a.limit})" if a.limit else ""))
    w("")

    for fam in [f.strip() for f in a.families.split(",") if f.strip()]:
        cfg = FAMILIES[fam]
        sets = build_sets(fam, cfg, a.out, a.limit)
        w("#" * 78)
        w(f" [{cfg['title']}]  test 구성: 배경 포함 {sets['with_bg']['n']:,}장 "
          f"(라벨 {sets['n_labeled']:,} + 배경 {sets['n_bg']:,}) / 배경 제외 {sets['no_bg']['n']:,}장")
        w("#" * 78)

        res = {}
        for mk, wt in cfg["models"].items():
            if not Path(wt).exists():
                w(f"  {mk}: 가중치 없음 ({wt}) - 건너뜀")
                continue
            print(f"[{fam}] {mk} 평가 중...", flush=True)
            res[mk] = dict(
                with_bg=evaluate(wt, sets["with_bg"]["yaml"], cfg["task"], a, f"{mk}_with_bg"),
                no_bg=evaluate(wt, sets["no_bg"]["yaml"], cfg["task"], a, f"{mk}_no_bg"),
                fp=background_fp(wt, sets["bg_only"]["list"], a),
            )
            w(f"  {mk:<10}: {wt}")
        if not res:
            w("")
            continue
        kinds = ["box", "mask"] if cfg["task"] == "segment" else ["box"]

        for kind in kinds:
            kname = "박스" if kind == "box" else "마스크"
            w("")
            w(f"■ 전체 성능 ({kname})")
            w(f"  {'모델':<12}{'':<24}| 배경 포함 test                  | 배경 제외 test")
            w(f"  {'':<36}| mAP50-95  mAP50     P       R   | mAP50-95  mAP50     P       R")
            for mk, r in res.items():
                cells = []
                for ts in ("with_bg", "no_bg"):
                    s = r[ts][kind]
                    cells.append(f" {s['map']:7.4f} {s['map50']:7.4f} {s['p']:7.4f} {s['r']:7.4f} ")
                    rows.append([fam, mk, ts, kind, "all", s["p"], s["r"], s["map50"], s["map"]])
                    for c, v in s["per"].items():
                        rows.append([fam, mk, ts, kind, c, v["p"], v["r"], v["map50"], v["map"]])
                w(f"  {mk:<12}{LABEL[mk]:<20}  |" + "|".join(cells))

        w("")
        w(f"■ 배경 사진 오탐 (test 배경 {sets['n_bg']}장, 물체가 없어야 정답)")
        w(f"  {'모델':<12}{'':<24}| 오탐 이미지(conf≥0.25) 검출 수 | 오탐 이미지(conf≥0.5) 검출 수 | 가장 많이 잘못 찾은 클래스")
        for mk, r in res.items():
            fp = r["fp"]
            top = ", ".join(f"{c} {k}" for c, k in fp["by_cls"].most_common(3)) or "없음"
            w(f"  {mk:<12}{LABEL[mk]:<20}  | {fp['img_hit'][0.25]:>4} / {fp['n']:<4} "
              f"({fp['img_hit'][0.25]/max(fp['n'],1)*100:5.1f}%) {fp['det'][0.25]:>6} "
              f"| {fp['img_hit'][0.5]:>4} / {fp['n']:<4} ({fp['img_hit'][0.5]/max(fp['n'],1)*100:5.1f}%) "
              f"{fp['det'][0.5]:>6} | {top}")
            for t in CONFS:
                rows.append([fam, mk, "bg_only", f"fp_conf{t}", "all",
                             None, None, fp["img_hit"][t], fp["det"][t]])

        for kind in kinds:
            kname = "박스" if kind == "box" else "마스크"
            mks = list(res)
            w("")
            w(f"■ 클래스별 성능 — 배경 제외 test, {kname} (차이 = 배경 포함 학습 - 배경 제외 학습)")
            head = f"  {'class':<9}" + "".join(f"| {mk:<10} P      R    mAP50-95 " for mk in mks)
            if len(mks) == 2:
                head += "| 차이 mAP50-95"
            w(head)
            classes = sorted({c for mk in mks for c in res[mk]["no_bg"][kind]["per"]})
            for c in classes:
                line = f"  {c:<9}"
                vals = []
                for mk in mks:
                    v = res[mk]["no_bg"][kind]["per"].get(c)
                    vals.append(v["map"] if v else None)
                    line += (f"| {'':<10}{v['p']:5.3f}  {v['r']:5.3f}  {v['map']:6.3f}  " if v
                             else f"| {'':<10}  -      -       -    ")
                if len(mks) == 2 and None not in vals:
                    line += f"|   {vals[0]-vals[1]:+.3f}"
                w(line)
        w("")

    w("=" * 78)
    w(" 읽는 법")
    w("=" * 78)
    w("- 배경 포함 test 는 배경 사진에서 오탐하면 정밀도(P)와 mAP 가 깎입니다.")
    w("  배경 제외 test 는 라벨된 물체를 찾는 성능만 봅니다.")
    w("- 배경 이미지가 효과가 있다면: 배경 사진 오탐이 줄고, 배경 포함 test 의 P 가 오르며,")
    w("  배경 제외 test 성능은 크게 떨어지지 않아야 합니다.")
    w("- 주의: 배경 제외 학습은 모델도 yolo26s(소형)로 달라서, 차이에는 배경 효과와 모델 크기 효과가")
    w("  함께 들어 있습니다. 배경만의 효과를 알려면 같은 모델로 배경을 넣고/빼고 학습한 결과가 필요합니다.")
    w("- 오탐 기준 conf 0.25 는 ultralytics 예측 기본값입니다. 실제 서비스 신뢰도 기준으로 다시 보세요.")
    w("- 검출과 세그는 분할이 서로 달라 test 이미지가 같지 않습니다. 검출 대 세그 숫자는 직접 비교하지 마세요.")
    w("- 세그 best.pt 는 박스 mAP50-95 + 마스크 mAP50-95 합이 가장 높은 에포크입니다.")

    txt = "\n".join(L) + "\n"
    (a.out / "compare_result.txt").write_text(txt, encoding="utf-8")
    with open(a.out / "compare_result.csv", "w", newline="", encoding="utf-8-sig") as f:
        cw = csv.writer(f)
        cw.writerow(["family", "model", "test_set", "metric", "class", "P", "R",
                     "mAP50_or_fp_images", "mAP50-95_or_fp_detections"])
        cw.writerows(rows)
    print(txt)
    print(f"저장: {a.out / 'compare_result.txt'}")


if __name__ == "__main__":
    main()
