"""중단 후 이어서 학습(resume)해 끊어진 기록을 하나로 합친다.

    uv run python 02_training/baseline_training/merge_resume_logs.py seg_yolo11n
    uv run python 02_training/baseline_training/merge_resume_logs.py --dir <결과 폴더>

이어서 학습하면 ultralytics 와 metrics_logger 가 시간을 0 부터 다시 센다.
그래서 세 파일의 시간 값이 재개 지점에서 끊긴다.

    results.csv   time 열 (누적 초)
    metrics.txt   누적 열
    summary.txt   총 소요 시간, 에포크 평균

이 스크립트는 에포크별 소요 시간을 이어 붙여 누적 값을 다시 계산하고,
중단·재개 기록을 summary.txt 에 추가한다. 원본은 *_raw 로 한 번만 보관하므로
여러 번 실행해도 결과가 같다. 중단된 시간(학습이 멈춰 있던 시간)은
학습 시간에 넣지 않고 따로 표시한다.
"""
# 현재 작업 디렉토리와 상관없이 공용 작업공간 경로를 찾는다.
from pathlib import Path as _WorkspacePath
import sys as _workspace_sys
_workspace_root = next(p for p in _WorkspacePath(__file__).resolve().parents if (p / 'workspace_paths.py').is_file())
_workspace_sys.path.insert(0, str(_workspace_root))
from workspace_paths import workspace_path


import csv
import re
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path

BASE = workspace_path('best_train/runs')
TS = "%Y-%m-%d %H:%M:%S"


def fmt(sec):
    return str(timedelta(seconds=int(round(sec))))


def raw_copy(p):
    """원본을 *_raw 로 한 번만 보관하고, 항상 원본을 읽는다."""
    raw = p.with_name(p.stem + "_raw" + p.suffix)
    if not raw.exists():
        shutil.copy2(p, raw)
    return raw


def main():
    if "--dir" in sys.argv:
        run = Path(sys.argv[sys.argv.index("--dir") + 1])
    else:
        run = workspace_path('best_train/runs', sys.argv[1] if len(sys.argv) > 1 else "seg_yolo11n")
    log = []
    say = lambda s: (print(s), log.append(s))
    say(f"[{datetime.now():{TS}}] 기록 통합 시작: {run}")

    # ── 1. metrics.txt: 에포크 행과 재개 구분선 읽기 ─────────────────────
    lines = raw_copy(run / "metrics.txt").read_text(encoding="utf-8").splitlines()
    row_re = re.compile(r"^\s*(\d+)\s+([\d.]+)s\s+(\S+)(.*)$")
    sep_re = re.compile(r"^---- (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) (\d+)에포크부터 이어서 학습")

    header, rows, resumes = [], [], []      # resumes: (재개 시각, 재개 에포크)
    for ln in lines:
        m, s = row_re.match(ln), sep_re.match(ln)
        if s:
            resumes.append((datetime.strptime(s.group(1), TS), int(s.group(2))))
        elif m:
            rows.append((int(m.group(1)), float(m.group(2)), m.group(4)))
        elif not rows:
            header.append(ln)
    if not rows:
        say("에포크 기록이 없어 중단합니다."); return
    resume_eps = {ep for _, ep in resumes}

    out, cum = list(header), 0.0
    for ep, dt, rest in rows:
        cum += dt
        tag = "  [재개]" if ep in resume_eps else ""
        out.append(f"{ep:>4} {dt:>7.1f}s {fmt(cum):>10}{rest}{tag}")
    if resumes:
        out.append("")
        out.append("※ 누적 시간은 에포크별 소요 시간을 이어 붙인 실제 학습 시간입니다.")
        out.append("  학습이 멈춰 있던 시간은 포함하지 않습니다.")
        for t, ep in resumes:
            out.append(f"  [재개] {ep}에포크: {t:{TS}} 에 이어서 학습 시작 "
                       f"(이 에포크 시간에는 재시작 준비 시간이 포함됨)")
    (run / "metrics.txt").write_text("\n".join(out) + "\n", encoding="utf-8")
    say(f"metrics.txt: {len(rows)}개 에포크, 재개 {len(resumes)}회, 누적 학습 시간 {fmt(cum)}")

    # ── 2. results.csv: time 열이 줄어드는 지점마다 앞 구간 끝값을 더한다 ──
    with open(raw_copy(run / "results.csv"), encoding="utf-8", newline="") as f:
        rd = list(csv.reader(f))
    head, data = rd[0], rd[1:]
    ti = next(i for i, c in enumerate(head) if c.strip() == "time")
    offset, prev, fixed = 0.0, 0.0, 0
    for r in data:
        t = float(r[ti])
        if t < prev:                 # 여기서 시간이 0 부터 다시 시작됨
            offset += prev
            fixed += 1
        prev = t
        r[ti] = f"{t + offset:.3f}"
    with open(run / "results.csv", "w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerows([head] + data)
    say(f"results.csv: 시간 초기화 {fixed}곳 보정, 마지막 누적 {fmt(float(data[-1][ti]))}")

    # ── 3. summary.txt: 시간 항목 교체 + 중단·재개 기록 추가 ─────────────
    cfg = (run / "config.txt").read_text(encoding="utf-8")
    start = datetime.strptime(re.search(r"시작 시각\s*:\s*(\S+ \S+)", cfg).group(1), TS)

    sp = run / "summary.txt"
    if not sp.exists():
        say("summary.txt 가 아직 없어 요약은 건너뜁니다 (학습이 끝나지 않음).")
        (run / "merge_log.txt").write_text("\n".join(log) + "\n", encoding="utf-8")
        return
    sm = raw_copy(sp).read_text(encoding="utf-8").splitlines()
    end_m = next((re.search(r"종료 시각\s*:\s*(\S+ \S+)", l) for l in sm if "종료 시각" in l), None)
    end = datetime.strptime(end_m.group(1), TS) if end_m else None

    dts = [dt for _, dt, _ in rows]
    steady = dts[1:] if len(dts) > 1 else dts
    mean = sum(steady) / len(steady)
    med = sorted(steady)[len(steady) // 2]

    # 구간별: [시작 시각, 시작 에포크] ~ 다음 재개 직전
    seg_starts = [(start, rows[0][0])] + resumes
    segs = []
    for i, (t0, ep0) in enumerate(seg_starts):
        ep1 = seg_starts[i + 1][1] - 1 if i + 1 < len(seg_starts) else rows[-1][0]
        train = sum(dt for ep, dt, _ in rows if ep0 <= ep <= ep1)
        segs.append((t0, ep0, ep1, train, t0 + timedelta(seconds=train)))

    block = ["", "-" * 62, " 중단·재개 기록", "-" * 62]
    for i, (t0, ep0, ep1, train, stop) in enumerate(segs):
        block.append(f"구간 {i+1}: {ep0}~{ep1}에포크  시작 {t0:{TS}}  "
                     f"학습 {fmt(train)}  (추정 종료 {stop:%H:%M:%S})")
        if i + 1 < len(segs):
            gap = (segs[i + 1][0] - stop).total_seconds()
            block.append(f"        -> 멈춰 있던 시간 약 {fmt(max(gap, 0))} "
                         f"(끝내지 못한 에포크 진행분 포함)")
    block.append("※ 추정 종료 = 구간 시작 시각 + 그 구간 에포크 시간의 합")

    new = []
    for l in sm:
        if l.startswith("총 소요 시간"):
            new.append(f"총 학습 시간    : {fmt(cum)}  (멈춰 있던 시간 제외, 에포크 시간 합)")
            if end and end > start:
                new.append(f"전체 경과 시간  : {fmt((end - start).total_seconds())}  "
                           f"({start:{TS}} ~ {end:{TS}}, 중단·최종 검증 포함)")
        elif l.startswith("에포크 평균"):
            new.append(f"에포크 평균     : {mean:.1f}초  (첫 에포크 제외, 중앙값 {med:.1f}초)")
        elif l.startswith("실행한 에포크"):
            new.append(l)
            if resumes:
                new.append(f"중단 후 재개    : {len(resumes)}회 "
                           f"({', '.join(f'{ep}에포크' for _, ep in resumes)}부터)")
        else:
            new.append(l)
    # 요약 끝의 '=' 줄 앞에 중단·재개 기록을 넣는다
    if resumes:
        cut = max(i for i, l in enumerate(new) if l.startswith("가중치:")) - 1
        new = new[:cut] + block + [""] + new[cut:]
    sp.write_text("\n".join(new) + "\n", encoding="utf-8")
    say(f"summary.txt: 총 학습 시간 {fmt(cum)}, 에포크 평균 {mean:.1f}초로 교체, 재개 기록 추가")
    say("원본은 metrics_raw.txt / results_raw.csv / summary_raw.txt 로 보관했습니다.")
    (run / "merge_log.txt").write_text("\n".join(log) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
