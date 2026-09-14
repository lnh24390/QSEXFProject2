"""학습 지표를 사람이 읽는 텍스트로 기록한다.

ultralytics 는 results.csv 를 남기지만 열 이름이 길고 클래스별 수치가 없다.
이 모듈은 콜백을 붙여 세 가지 파일을 추가로 남긴다.

    metrics.txt      에포크별 한 줄 표 (진행 중에도 바로 열어볼 수 있다)
    summary.txt      학습 종료 후 요약 + 클래스별 성능 + 조기 종료 여부
    config.txt       학습 환경과 설정 (나중에 실행끼리 비교할 때 쓴다)

사용법:

    from metrics_logger import attach
    model = YOLO("yolo11n.pt")
    attach(model)
    model.train(...)
"""

import csv
import os
import time
from datetime import datetime, timedelta


def _fmt(sec):
    """초를 1시간 23분 45초 형태로."""
    return str(timedelta(seconds=int(sec)))


def attach(model, note=""):
    """model 에 기록용 콜백을 붙인다. note 는 config.txt 에 남길 메모."""
    state = {"t0": None, "prev": 0.0, "best": -1.0, "best_epoch": 0, "seen": set()}

    # ── 학습 시작: 환경과 설정을 기록 ──────────────────────────────────
    def on_start(tr):
        import torch
        state["t0"] = time.time()
        d = tr.save_dir
        os.makedirs(d, exist_ok=True)

        # 중단된 학습을 이어서 할 때(resume)는 기존 기록을 지우지 않고 이어 쓴다.
        if getattr(tr, "start_epoch", 0) > 0 and (d / "metrics.txt").exists():
            try:
                with open(d / "results.csv", encoding="utf-8") as stream:
                    rows = list(csv.DictReader(stream))
                k = {c.strip(): c for c in rows[0]}
                for r in rows:
                    v = float(r[k["metrics/mAP50-95(B)"]]) + (float(r[k["metrics/mAP50-95(M)"]]) if "metrics/mAP50-95(M)" in k else 0)
                    if v > state["best"]:
                        state["best"], state["best_epoch"] = v, int(r[k["epoch"]])
                state["prev"] = 0.0
            except Exception:
                pass
            with open(d / "metrics.txt", "a", encoding="utf-8") as f:
                f.write(f"---- {datetime.now():%Y-%m-%d %H:%M:%S} {tr.start_epoch + 1}에포크부터 이어서 학습 "
                        f"(누적 시간은 재시작 시점부터 다시 셉니다) ----\n")
            with open(d / "config.txt", "a", encoding="utf-8") as f:
                f.write(f"\n이어서 학습 시작: {datetime.now():%Y-%m-%d %H:%M:%S}, "
                        f"{tr.start_epoch + 1}에포크부터\n")
            return

        n = {s: 0 for s in ("train", "val")}
        bg = {s: 0 for s in ("train", "val")}
        try:
            # 배경(빈 라벨) 이미지가 몇 장인지 세어 둔다. 오탐 억제 효과를
            # 나중에 해석하려면 이 수치가 필요하다.
            for s, ds in (("train", tr.train_loader.dataset),
                          ("val", tr.test_loader.dataset)):
                n[s] = len(ds.labels)
                bg[s] = sum(1 for L in ds.labels if len(L["cls"]) == 0)
        except Exception:
            pass

        with open(d / "config.txt", "w", encoding="utf-8") as f:
            w = lambda s: f.write(s + "\n")
            w("=" * 62)
            w(" 학습 설정")
            w("=" * 62)
            if note:
                w(f"메모            : {note}")
            w(f"시작 시각       : {datetime.now():%Y-%m-%d %H:%M:%S}")
            w(f"모델            : {tr.args.model}")
            w(f"데이터          : {tr.args.data}")
            w(f"이미지 크기     : {tr.args.imgsz}")
            auto = isinstance(tr.args.batch, float) and 0 < tr.args.batch < 1
            w(f"batch           : {getattr(tr, 'batch_size', tr.args.batch)}"
              + (f"  (자동 batch: GPU 메모리 {tr.args.batch*100:.0f}% 기준으로 정함)" if auto
                 else "  (자동 batch: GPU 메모리 60% 기준으로 정함)" if tr.args.batch == -1 else ""))
            w(f"workers         : {tr.args.workers}")
            w(f"cache           : {tr.args.cache}")
            w(f"AMP             : {tr.args.amp}")
            w(f"epochs (최대)   : {tr.args.epochs}")
            w(f"patience(조기종료): {tr.args.patience}")
            w(f"optimizer       : {tr.args.optimizer}")
            w(f"seed            : {tr.args.seed}")
            w("-" * 62)
            w(f"학습 이미지     : {n['train']:,} (배경 {bg['train']:,}장, "
              f"{bg['train']/max(n['train'],1)*100:.1f}%)")
            w(f"검증 이미지     : {n['val']:,} (배경 {bg['val']:,}장, "
              f"{bg['val']/max(n['val'],1)*100:.1f}%)")
            w(f"클래스 {len(tr.data['names'])}개    : "
              f"{', '.join(tr.data['names'].values())}")
            w("-" * 62)
            w(f"device          : {tr.args.device}")
            w(f"torch           : {torch.__version__}")
            if torch.cuda.is_available():
                w(f"GPU             : {torch.cuda.get_device_name(0)} "
                  f"({torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB)")
            w("=" * 62)

        hdr = (f"{'ep':>4} {'시간':>8} {'누적':>10} " +
               ' '.join(f'{name:>9}' for name in tr.loss_names) + ' ' +
               f"{'P':>7} {'R':>7} {'mAP50':>7} {'mAP50-95':>9} "
               f"{'GPU_GB':>7} {'lr':>9}  비고")
        with open(d / "metrics.txt", "w", encoding="utf-8") as f:
            f.write(hdr + "\n" + "-" * len(hdr) + "\n")

    # ── 에포크마다: 한 줄 추가 ─────────────────────────────────────────
    def on_epoch(tr):
        import torch
        d = tr.save_dir
        m = tr.metrics or {}

        # 학습이 끝난 뒤 돌리는 '최종 검증(best.pt)' 도 이 콜백을 부른다.
        # 그것은 에포크가 아니므로 번호를 매기지 않고 따로 표시한다.
        # results.csv 는 실제 에포크마다 한 줄씩만 쌓이므로 이것으로 구분한다.
        try:
            with open(d / "results.csv", encoding="utf-8") as stream:
                n_rows = sum(1 for _ in stream) - 1
        except Exception:
            n_rows = tr.epoch + 1
        is_final = (tr.epoch + 1) > n_rows
        if is_final and state["seen"]:
            return              # 최종 검증은 summary.txt 에서 다루므로 표에는 넣지 않는다
        state["seen"].add(tr.epoch)
        el = time.time() - state["t0"]
        dt, state["prev"] = el - state["prev"], el

        g = lambda k: float(m.get(k, 0) or 0)
        cur = g("metrics/mAP50-95(B)") + g("metrics/mAP50-95(M)")
        mark = ""
        if cur > state["best"]:
            state["best"], state["best_epoch"] = cur, tr.epoch + 1
            mark = "<- 최고"

        lo = tr.label_loss_items(tr.tloss)
        loss_text = " ".join(f"{float(lo['train/' + name]):>9.4f}" for name in tr.loss_names)
        gb = torch.cuda.max_memory_reserved()/1e9 if torch.cuda.is_available() else 0

        with open(d / "metrics.txt", "a", encoding="utf-8") as f:
            f.write(f"{tr.epoch+1:>4} {dt:>7.1f}s {_fmt(el):>10} "
                    f"{loss_text} "
                    f"{g('metrics/precision(B)'):>7.4f} {g('metrics/recall(B)'):>7.4f} "
                    f"{g('metrics/mAP50(B)'):>7.4f} {g('metrics/mAP50-95(B)'):>9.4f} "
                    f"{gb:>7.2f} {tr.optimizer.param_groups[0]['lr']:>9.6f}  {mark}\n")

    # ── 학습 종료: 요약 ────────────────────────────────────────────────
    def on_end(tr):
        d = tr.save_dir
        el = time.time() - state["t0"]
        done = max(state["seen"], default=tr.epoch) + 1
        stopped_early = done < tr.args.epochs

        rows = []
        try:
            with open(d / "results.csv", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
        except Exception:
            pass

        with open(d / "summary.txt", "w", encoding="utf-8") as f:
            w = lambda s: f.write(s + "\n")
            w("=" * 62)
            w(" 학습 결과 요약")
            w("=" * 62)
            w(f"종료 시각       : {datetime.now():%Y-%m-%d %H:%M:%S}")
            w(f"총 소요 시간    : {_fmt(el)}")
            w(f"실행한 에포크   : {done} / {tr.args.epochs} (최대)")
            if done > 1:
                w(f"현재 실행 구간 평균: {el/max(len(state['seen']),1):.1f}초 "
                  f"(준비·종료 평가 포함; 순수 에포크 시간은 results.csv 확인)")
            w("")
            if stopped_early:
                w(f"■ 조기 종료됨. patience={tr.args.patience} 에포크 동안 "
                  f"모델 선택 점수가 나아지지 않았습니다.")
                w(f"  최고 성능은 {state['best_epoch']} 에포크였고, "
                  f"이후 {done - state['best_epoch']} 에포크 더 돌린 뒤 멈췄습니다.")
                w(f"  남은 {tr.args.epochs - done} 에포크만큼 시간을 아꼈습니다.")
            else:
                w(f"■ 최대 에포크({tr.args.epochs})까지 모두 실행했습니다.")
                w(f"  조기 종료가 걸리지 않았으므로 더 학습하면 나아질 여지가 "
                  f"있을 수 있습니다.")
            w("")
            w("선택 기준: 검출은 박스 mAP50–95, 분할은 박스+마스크 mAP50–95 합")
            w(f"최고 선택 점수  : {state['best']:.4f}  ({state['best_epoch']} 에포크)")
            w("")
            w("-" * 62)
            w(" 클래스별 성능 (검증 자료, 최고 가중치 기준)")
            w("-" * 62)
            try:
                b = tr.validator.metrics.box
                w(f"{'클래스':<10}{'정밀도':>9}{'재현율':>9}{'mAP50':>9}{'mAP50-95':>10}")
                for i, c in enumerate(b.ap_class_index):
                    nm = tr.data["names"][int(c)]
                    w(f"{nm:<10}{b.p[i]:>9.3f}{b.r[i]:>9.3f}"
                      f"{b.ap50[i]:>9.3f}{b.ap[i]:>10.3f}")
                w("")
                mask = getattr(tr.validator.metrics, 'seg', None)
                if mask is not None:
                    w("마스크 클래스별 성능 (정밀도 / 재현율 / mAP50 / mAP50–95)")
                    for i, c in enumerate(mask.ap_class_index):
                        nm = tr.data['names'][int(c)]
                        w(f"{nm:<10}{mask.p[i]:>9.3f}{mask.r[i]:>9.3f}{mask.ap50[i]:>9.3f}{mask.ap[i]:>10.3f}")
                w("재현율이 낮은 클래스는 놓치는 물체가 많다는 뜻입니다.")
                w("정밀도가 낮으면 없는 물체를 있다고 하는 오탐이 많다는 뜻입니다.")
            except Exception as e:
                w(f"(클래스별 수치를 읽지 못했습니다: {e})")
            w("")
            w("-" * 62)
            w(" 마지막 5개 에포크")
            w("-" * 62)
            if rows:
                k = {c.strip(): c for c in rows[0]}
                w(f"{'ep':>4}{'mAP50':>9}{'mAP50-95':>10}")
                for r in rows[-5:]:
                    w(f"{r[k['epoch']]:>4}"
                      f"{float(r[k['metrics/mAP50(B)']]):>9.4f}"
                      f"{float(r[k['metrics/mAP50-95(B)']]):>10.4f}")
            w("")
            w("=" * 62)
            w(f"가중치: {d}\\weights\\best.pt")
            w(f"에포크별 상세: {d}\\metrics.txt")
            w("=" * 62)

    def on_structured_end(tr):
        from pathlib import Path
        import sys
        root = next(p for p in Path(__file__).resolve().parents if (p/'workspace_paths.py').is_file())
        sys.path[:0] = [str(root), str(root/'02_training')]
        from workspace_paths import workspace_path
        from model_analysis.history import training_summary
        from model_analysis.schema import unique_dir, write_json
        value = training_summary(tr.save_dir)
        if value:
            folder = unique_dir(workspace_path('model_analysis_results')/'training'/'baseline_training', tr.save_dir.name)
            write_json(folder/'summary.json', value)

    model.add_callback("on_train_start", on_start)
    model.add_callback("on_fit_epoch_end", on_epoch)
    model.add_callback("on_train_end", on_end)
    model.add_callback("on_train_end", on_structured_end)
    return model
