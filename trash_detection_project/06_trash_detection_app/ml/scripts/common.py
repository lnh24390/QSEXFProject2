"""스크립트 공통 유틸: 경로 상수, 로깅, 앱 설정 파일 접근."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# Windows 콘솔 한글 깨짐 방지
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

ML_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = ML_DIR.parent
APP_DIR = ROOT_DIR / "app"
APP_MODELS_DIR = APP_DIR / "assets" / "models"
APP_GUIDE_JSON = APP_DIR / "assets" / "config" / "recycling_guide.json"
WEIGHTS_DIR = ML_DIR / "weights"
RUNS_DIR = ML_DIR / "runs"
EXPORTS_DIR = ML_DIR / "exports"
DATASETS_DIR = ML_DIR / "datasets"
CONFIGS_DIR = ML_DIR / "configs"


def info(msg: str) -> None:
    print(f"[ml] {msg}", flush=True)


def warn(msg: str) -> None:
    print(f"[ml][경고] {msg}", file=sys.stderr, flush=True)


def die(msg: str, code: int = 1) -> None:
    print(f"[ml][오류] {msg}", file=sys.stderr, flush=True)
    sys.exit(code)


def normalize_class_name(name: str) -> str:
    """앱(recycling_guide.dart)과 동일한 정규화 규칙."""
    s = name.strip().lower()
    s = re.sub(r"[\s\-&/]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")


def load_guide() -> dict:
    with open(APP_GUIDE_JSON, encoding="utf-8") as f:
        return json.load(f)


def resolve_guide_category(guide: dict, class_name: str) -> tuple[str, str]:
    """앱과 같은 순서로 카테고리를 찾는다. (카테고리 id, 'ignored'|'explicit'|'rule'|'fallback')"""
    key = normalize_class_name(class_name)
    if key in {normalize_class_name(n) for n in guide.get("ignore", [])}:
        return "-", "ignored"
    classes = {normalize_class_name(k): v for k, v in guide.get("classes", {}).items()}
    if key in classes:
        return classes[key], "explicit"
    for rule in guide.get("rules", []):
        if all(kw in key for kw in rule["contains"]):
            return rule["category"], "rule"
    return guide.get("fallback", "general"), "fallback"


def model_class_names(weights: Path) -> list[str]:
    """.pt / .onnx / .tflite 에서 클래스명 목록을 읽는다."""
    suffix = weights.suffix.lower()
    if suffix == ".pt":
        from ultralytics import YOLO

        names = YOLO(str(weights)).names
        return [names[i] for i in sorted(names)]
    if suffix == ".onnx":
        import ast

        import onnx

        model = onnx.load(str(weights))
        meta = {p.key: p.value for p in model.metadata_props}
        if "names" in meta:
            names = ast.literal_eval(meta["names"])
            return [names[i] for i in sorted(names)]
        return []
    if suffix == ".tflite":
        # Ultralytics 는 tflite 파일 끝에 metadata.json(구버전은 metadata.yaml) 을 담은 zip 을 덧붙인다.
        # zipfile 은 앞에 다른 데이터가 붙은 zip 도 EOCD 로 찾아 열 수 있다.
        import json as _json
        import zipfile

        import yaml

        try:
            with zipfile.ZipFile(weights) as zf:
                for n in zf.namelist():
                    if n.endswith(("metadata.json", "metadata.yaml", "metadata.yml")):
                        raw = zf.read(n).decode("utf-8")
                        meta = _json.loads(raw) if n.endswith(".json") else yaml.safe_load(raw)
                        names = meta.get("names", {})
                        if isinstance(names, dict):
                            return [names[k] for k in sorted(names, key=int)]
                        return list(names)
        except (zipfile.BadZipFile, ValueError, KeyError):
            return []
    return []
