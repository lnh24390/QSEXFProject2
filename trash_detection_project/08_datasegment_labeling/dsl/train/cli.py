"""Entry point of the training subprocess.

    python -m dsl.train.cli runs/<name>/job.json
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

from .jobs import TrainJob
from .registry import get as get_spec

TRAINERS = {
    "ultralytics": "dsl.train.trainer_ultralytics",
    "torchvision": "dsl.train.trainer_maskrcnn",
    "hf": "dsl.train.trainer_hfseg",
}


def main(argv) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, OSError):
            pass

    if len(argv) < 1:
        print("사용법: python -m dsl.train.cli <job.json>")
        return 2

    job_path = Path(argv[0])
    try:
        job = TrainJob.load(job_path).fill_defaults()
    except (OSError, ValueError, TypeError) as e:
        print(f"[DSL] 잡 파일을 읽지 못했습니다: {e}")
        return 2

    spec = get_spec(job.model_key)
    if spec is None:
        print(f"[DSL] 알 수 없는 모델 키: {job.model_key}")
        return 2

    module_name = TRAINERS.get(spec.framework)
    if module_name is None:
        print(f"[DSL] 지원하지 않는 프레임워크: {spec.framework}")
        return 2

    print(f"[DSL] === {spec.display} 학습 시작 ===")
    print(f"[DSL] {job.describe()}")
    try:
        import importlib
        mod = importlib.import_module(module_name)
        summary = mod.train(job)
    except SystemExit as e:
        print(f"[DSL] 중단: {e}")
        return 1
    except KeyboardInterrupt:
        print("[DSL] 사용자 요청으로 중단되었습니다.")
        return 130
    except Exception:
        print("[DSL] 학습 중 오류가 발생했습니다:")
        traceback.print_exc()
        return 1

    out = job.run_dir / "summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"[DSL] === 완료 ===")
    print(f"[DSL] 결과: {summary.get('save_dir', job.run_dir)}")
    if summary.get("best"):
        print(f"[DSL] 최적 가중치: {summary['best']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
