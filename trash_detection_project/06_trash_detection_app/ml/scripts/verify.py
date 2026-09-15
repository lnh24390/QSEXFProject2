"""변환된 모델(.tflite / .onnx) 검증: 입출력 텐서 확인 + 샘플 이미지 추론.

사용 예)
  python scripts/verify.py --model exports/best.tflite --image sample.jpg
  python scripts/verify.py --model exports/best.onnx  --image sample.jpg

.tflite 추론은 tensorflow 또는 ai-edge-litert 가 설치된 환경에서만 됩니다(Windows 에서는 보통 미설치).
그 경우 --docker 를 붙이면 Docker 컨테이너에서 실행합니다.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

from common import ML_DIR, die, info, model_class_names, warn


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--image", type=Path, help="테스트 이미지. 없으면 난수 입력으로 shape 만 검사")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--docker", action="store_true")
    return p.parse_args()


def letterbox(img, size: int):
    from PIL import Image

    w, h = img.size
    scale = size / max(w, h)
    nw, nh = round(w * scale), round(h * scale)
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    canvas.paste(img.resize((nw, nh)), ((size - nw) // 2, (size - nh) // 2))
    return canvas


def decode_raw(output: np.ndarray, names: list[str], conf: float, size: int):
    """raw 출력 [1, 4+nc, N] 또는 [1, N, 4+nc] 을 (cls, conf, xyxy) 로 디코딩 (NMS 없음)."""
    out = output[0]
    if out.shape[0] < out.shape[1]:
        out = out.T  # [N, 4+nc]
    boxes, scores = out[:, :4], out[:, 4:]
    cls = scores.argmax(1)
    sc = scores.max(1)
    keep = sc >= conf
    # 좌표가 0~1 정규화(LiteRT) 인지 픽셀 단위(ONNX) 인지 자동 판별
    if boxes.max() <= 1.5:
        boxes = boxes * size
    dets = []
    for b, c, s in zip(boxes[keep], cls[keep], sc[keep]):
        cx, cy, w, h = b
        dets.append((names[c] if c < len(names) else str(c), float(s), (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)))
    dets.sort(key=lambda d: -d[1])
    return dets


def run_tflite(model: Path, image: Path | None, conf: float):
    interp = None
    try:
        from ai_edge_litert.interpreter import Interpreter  # type: ignore

        interp = Interpreter(model_path=str(model))
    except ImportError:
        try:
            import tensorflow as tf  # type: ignore

            interp = tf.lite.Interpreter(model_path=str(model))
        except ImportError:
            die("tensorflow 또는 ai-edge-litert 가 없습니다. --docker 로 실행하세요.")
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    outs = interp.get_output_details()
    info(f"input : {inp['shape']} {inp['dtype'].__name__}")
    for o in outs:
        info(f"output: {o['shape']} {o['dtype'].__name__}")
    _, h, w, _ = inp["shape"]
    x = prepare_input(image, w, inp["dtype"], inp.get("quantization", (0.0, 0)))
    interp.set_tensor(inp["index"], x)
    interp.invoke()
    out = interp.get_tensor(outs[0]["index"]).astype(np.float32)
    q = outs[0].get("quantization", (0.0, 0))
    if q[0]:
        out = (out - q[1]) * q[0]
    return out, w


def run_onnx(model: Path, image: Path | None, conf: float):
    import onnxruntime as ort

    sess = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    info(f"input : {inp.shape} {inp.type}")
    for o in sess.get_outputs():
        info(f"output: {o.shape} {o.type}")
    size = int(inp.shape[2])
    x = prepare_input(image, size, np.float32, (0.0, 0), nchw=True)
    out = sess.run(None, {inp.name: x})[0]
    return out, size


def prepare_input(image: Path | None, size: int, dtype, quant, nchw: bool = False):
    if image is None:
        warn("이미지가 없어 난수 입력으로 shape 만 검사합니다.")
        arr = np.random.rand(size, size, 3).astype(np.float32)
    else:
        from PIL import Image

        arr = np.asarray(letterbox(Image.open(image).convert("RGB"), size), dtype=np.float32) / 255.0
    if nchw:
        arr = arr.transpose(2, 0, 1)
    arr = arr[None]
    if dtype == np.int8 or dtype == np.uint8:
        scale, zp = quant
        arr = np.clip(np.round(arr / scale + zp), np.iinfo(dtype).min, np.iinfo(dtype).max).astype(dtype)
    return arr.astype(dtype) if dtype in (np.float32, np.float16) else arr


def main() -> None:
    args = parse_args()
    if not args.model.exists():
        die(f"모델이 없습니다: {args.model}")

    if args.docker:
        rel = args.model.resolve().relative_to(ML_DIR).as_posix()
        cmd = f"pip -q install pillow && python scripts/verify.py --model {rel} --conf {args.conf}"
        if args.image:
            cmd += f" --image {args.image.resolve().relative_to(ML_DIR).as_posix()}"
        r = subprocess.run(["docker", "run", "--rm", "-v", f"{ML_DIR.resolve().as_posix()}:/work", "-w", "/work",
                            "ultralytics/ultralytics:latest-cpu", "bash", "-c", cmd])
        sys.exit(r.returncode)

    names = model_class_names(args.model)
    info(f"클래스 {len(names)}개: {names[:10]}{' …' if len(names) > 10 else ''}")

    if args.model.suffix == ".tflite":
        out, size = run_tflite(args.model, args.image, args.conf)
    elif args.model.suffix == ".onnx":
        out, size = run_onnx(args.model, args.image, args.conf)
    else:
        die("지원 형식: .tflite, .onnx")

    info(f"raw output shape: {out.shape}")
    if args.image:
        dets = decode_raw(out, names, args.conf, size)
        info(f"검출 {len(dets)}개 (NMS 전, conf>={args.conf}):")
        for name, s, box in dets[:15]:
            info(f"  {name:20s} {s:.2f}  {tuple(round(v) for v in box)}")


if __name__ == "__main__":
    main()
