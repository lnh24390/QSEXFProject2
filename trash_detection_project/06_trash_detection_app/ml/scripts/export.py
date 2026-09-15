"""학습된 .pt → 다양한 배포 포맷으로 변환 (LiteRT/TFLite, ONNX, TorchScript, OpenVINO, NCNN …).

Android(ultralytics_yolo 플러그인)는 `format=litert nms=None` 으로 만든 .tflite 가 필요합니다.
LiteRT 변환은 Linux x86 / macOS 에서만 되므로 Windows 에서는 Docker 로 자동 전환합니다
(Docker Desktop 실행 필요, ultralytics/ultralytics:latest-cpu 이미지 사용). 나머지 포맷은 Windows 에서 바로 됩니다.

양자화(INT8 / FP16 등) 변환은 사용하지 않습니다(비활성화). 모든 결과물은 FP32 입니다.

지원 포맷 (--format, 콤마로 여러 개):
  litert       Android 앱용 .tflite (FP32)                          [Windows: Docker]
  onnx         ONNX (onnxruntime, 서버/데스크톱, 다른 툴체인의 입력)
  torchscript  PyTorch 배포용 .torchscript
  openvino     Intel CPU/iGPU 용 폴더
  ncnn         모바일 CPU(ncnn) 용 폴더
  all          위 전부

사용 예)
  # Android 앱용 변환 후 앱 assets/models 에 복사 (Windows 는 Docker 자동 사용)
  python scripts/export.py --weights runs/trash_yolo26n/weights/best.pt --install-to-app --app-name trash_v1

  # 모든 포맷 한 번에
  python scripts/export.py --weights best.pt --format all

  # 이미 다른 도구로 만든 .tflite 를 앱에 넣기
  python scripts/export.py --tflite my_model.tflite --install-to-app --app-name trash_v2
"""
from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from common import (
    APP_MODELS_DIR,
    EXPORTS_DIR,
    ML_DIR,
    die,
    info,
    load_guide,
    model_class_names,
    resolve_guide_category,
    warn,
)

DOCKER_IMAGE = "ultralytics/ultralytics:latest-cpu"

# 포맷명 → (ultralytics format, 추가 인자, 결과 파일명 접미사)
# 양자화 비활성화: half / int8 / quantize 인자를 넘기는 포맷은 두지 않는다.
NATIVE_FORMATS: dict[str, tuple[str, dict, str | None]] = {
    "onnx": ("onnx", {"simplify": True}, None),
    "torchscript": ("torchscript", {}, None),
    "openvino": ("openvino", {}, None),
    "ncnn": ("ncnn", {}, None),
}
ALL_FORMATS = ["litert", *NATIVE_FORMATS]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", type=Path, help="학습된 .pt 파일")
    p.add_argument("--tflite", type=Path, help="이미 변환된 .tflite 를 앱에 설치만 할 때")
    p.add_argument("--format", default="litert",
                   help="콤마 구분: " + ", ".join(ALL_FORMATS) + ", all (기본 litert)")
    p.add_argument("--imgsz", type=int, default=640, help="입력 해상도(학습값과 동일하게)")
    p.add_argument("--docker", action="store_true", help="Docker 컨테이너에서 변환 (Windows 필수)")
    p.add_argument("--docker-image", default=DOCKER_IMAGE)
    p.add_argument("--install-to-app", action="store_true", help="결과 .tflite 를 app/assets/models 에 복사")
    p.add_argument("--app-name", default=None, help="앱에 복사할 파일명(확장자 제외). 기본: 가중치 stem")
    p.add_argument("--check-guide", action="store_true", default=True,
                   help="모델 클래스가 recycling_guide.json 에 매핑돼 있는지 검사(기본 켜짐)")
    return p.parse_args()


def to_ml_relative(path: Path) -> Path:
    """ml/ 폴더 기준 상대 경로 (Docker 마운트용). ml/ 바깥이면 weights/ 로 복사."""
    path = path.resolve()
    try:
        return path.relative_to(ML_DIR)
    except ValueError:
        dst = ML_DIR / "weights" / path.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dst)
        warn(f"{path} 는 ml/ 밖에 있어 {dst} 로 복사했습니다.")
        return dst.relative_to(ML_DIR)


def find_new_tflite(search_dir: Path, since: float) -> Path | None:
    cands = [p for p in search_dir.rglob("*.tflite") if p.stat().st_mtime >= since - 1]
    if not cands:
        return None
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0]


def export_litert_native(weights: Path, args: argparse.Namespace) -> Path:
    from ultralytics import YOLO

    kwargs = dict(format="litert", nms=None, imgsz=args.imgsz)
    info(f"LiteRT 변환(native, FP32): {weights} {kwargs}")
    out = YOLO(str(weights)).export(**kwargs)
    out = Path(out)
    if out.is_dir():
        found = find_new_tflite(out, 0)
        if not found:
            die(f"변환 결과 폴더에 .tflite 가 없습니다: {out}")
        return found
    return out


def export_litert_docker(weights: Path, args: argparse.Namespace) -> Path:
    if shutil.which("docker") is None:
        die("docker 명령을 찾을 수 없습니다. Docker Desktop 을 설치/실행하세요.")
    if subprocess.run(["docker", "info"], capture_output=True).returncode != 0:
        die("Docker 데몬이 실행 중이 아닙니다. Docker Desktop 을 켠 뒤 다시 시도하세요.")

    rel = to_ml_relative(weights).as_posix()
    # 컨테이너 이미지에 LiteRT 변환 패키지가 없어 ultralytics 가 런타임 중 자동 설치하면
    # torch 불일치 오류가 나므로, 먼저 설치한 뒤 새 프로세스에서 export 한다.
    cmd_in = (
        "pip install -q 'litert-torch>=0.9.0' 'ai-edge-litert>=2.1.4' && "
        f"yolo export model={rel} format=litert nms=None imgsz={args.imgsz}"
    )

    import time

    started = time.time()
    docker_cmd = [
        "docker", "run", "--rm",
        "-v", f"{ML_DIR.resolve().as_posix()}:/work",
        "-w", "/work",
        args.docker_image,
        "bash", "-c", cmd_in,
    ]
    info("Docker 변환: " + " ".join(docker_cmd))
    r = subprocess.run(docker_cmd)
    if r.returncode != 0:
        die("Docker 변환 실패. 위 로그를 확인하세요.")

    search_dir = (ML_DIR / rel).parent
    found = find_new_tflite(search_dir, started)
    if not found:
        die(f"변환된 .tflite 를 {search_dir} 에서 찾지 못했습니다.")
    return found


def export_native(fmt: str, weights: Path, args: argparse.Namespace) -> Path:
    """Windows 포함 어디서나 되는 포맷을 변환해 exports/ 에 복사한다."""
    from ultralytics import YOLO

    real, extra, suffix = NATIVE_FORMATS[fmt]
    info(f"{fmt} 변환: {weights}")
    out = Path(YOLO(str(weights)).export(format=real, imgsz=args.imgsz, **extra))
    if suffix:
        out = out.rename(out.with_name(weights.stem + suffix))
    dst = EXPORTS_DIR / out.name
    if out.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(out, dst)
    else:
        shutil.copy2(out, dst)
    info(f"{fmt} 저장: {dst}")
    return dst


def check_guide(names: list[str]) -> None:
    if not names:
        warn("모델에서 클래스명을 읽지 못해 매핑 검사를 건너뜁니다.")
        return
    guide = load_guide()
    info(f"recycling_guide.json 매핑 검사 ({len(names)} 클래스)")
    for n in names:
        cat, how = resolve_guide_category(guide, n)
        mark = {"explicit": "  ", "rule": "~ ", "fallback": "! ", "ignored": "- "}[how]
        info(f"  {mark}{n:28s} → {cat:10s} ({how})")
    fallbacks = [n for n in names if resolve_guide_category(guide, n)[1] == "fallback"]
    if fallbacks:
        warn(f"기본값(general)으로 떨어지는 클래스 {len(fallbacks)}개: {fallbacks}")
        warn("app/assets/config/recycling_guide.json 의 classes 에 추가하세요.")


def install_to_app(tflite: Path, app_name: str | None) -> Path:
    APP_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    name = (app_name or tflite.stem) + ".tflite"
    dst = APP_MODELS_DIR / name
    shutil.copy2(tflite, dst)
    info(f"앱에 설치됨: {dst}")
    info("앱의 기본 모델로 쓰려면 app/lib/config/app_config.dart 의 defaultModelAsset 을 "
         f"'assets/models/{name}' 로 바꾸세요. (앱 내 모델 선택 메뉴에는 자동으로 나타납니다)")
    return dst


def main() -> None:
    args = parse_args()
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.tflite:
        if not args.tflite.exists():
            die(f"파일이 없습니다: {args.tflite}")
        if args.check_guide:
            check_guide(model_class_names(args.tflite))
        if args.install_to_app:
            install_to_app(args.tflite, args.app_name)
        else:
            warn("--install-to-app 가 없어 아무것도 하지 않았습니다.")
        return

    if not args.weights:
        die("--weights 또는 --tflite 를 지정하세요.")
    if not args.weights.exists():
        die(f"가중치 파일이 없습니다: {args.weights}")

    names = model_class_names(args.weights)
    info(f"클래스 {len(names)}개: {names}")

    formats = [f.strip() for f in args.format.split(",") if f.strip()]
    if "all" in formats:
        formats = list(ALL_FORMATS)
    unknown = [f for f in formats if f not in ALL_FORMATS]
    if unknown:
        die(f"지원하지 않는 포맷: {unknown}. 가능: {ALL_FORMATS}, all (양자화 포맷은 비활성화됨)")

    produced: list[Path] = []
    failed: list[str] = []
    for fmt in formats:
        if fmt == "litert":
            continue
        try:
            produced.append(export_native(fmt, args.weights, args))
        except Exception as e:  # noqa: BLE001
            warn(f"{fmt} 변환 실패: {e}")
            failed.append(fmt)

    tflite_path: Path | None = None
    if "litert" in formats:
        is_windows = platform.system() == "Windows"
        if args.docker or is_windows:
            if is_windows and not args.docker:
                warn("Windows 는 LiteRT 네이티브 변환을 지원하지 않아 --docker 로 전환합니다.")
            tflite_path = export_litert_docker(args.weights, args)
        else:
            tflite_path = export_litert_native(args.weights, args)
        dst = EXPORTS_DIR / tflite_path.name
        if tflite_path.resolve() != dst.resolve():
            shutil.copy2(tflite_path, dst)
        info(f"TFLite 저장: {dst}")
        produced.append(dst)
        if args.check_guide:
            check_guide(names)
        if args.install_to_app:
            install_to_app(dst, args.app_name)

    info("완료: " + ", ".join(str(p) for p in produced))
    if failed:
        warn(f"실패한 포맷: {failed}")
    if tflite_path and not args.install_to_app:
        info("앱에 넣으려면 --install-to-app 옵션을 추가하세요.")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    sys.exit(main())
