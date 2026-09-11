from collections import Counter
import json
import math
import os
import random
import time
from pathlib import Path

import cv2
from IPython.display import display
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
import torch
import yaml
from ultralytics import YOLO

if __name__ == '__main__':
    # =========================================================
    # 1. GPU / PyTorch 실행 환경 및 경로 최적화
    # =========================================================
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        DEVICE = 0
        GPU_NAME = torch.cuda.get_device_name(0)
        WORKERS = 4
        print(f"GPU 연결 성공: {GPU_NAME} (Device: {DEVICE})")
    else:
        DEVICE = "cpu"
        WORKERS = 0
        print("경고: CUDA 가능한 GPU를 찾을 수 없어 CPU 모드로 동작합니다.")

    print("PyTorch 버전:", torch.__version__)

    PROJECT = Path(r"D:/QSEXFProject2")
    DATASET_NAME = "YOLO_7CLASS_10000_PER_CLASS_20260909"
    DATASET = PROJECT / DATASET_NAME
    MODEL_NAME = "yolo11n.pt"

    EPOCHS = 50
    IMGSZ = 640
    BATCH = 8
    SMOKE_TEST = False

    NAMES = ["paper", "plastic", "can", "glass", "battery", "vinyl", "general"]
    OUTPUT = PROJECT / "yolo11_output" / DATASET_NAME
    OUTPUT.mkdir(parents=True, exist_ok=True)

    assert (DATASET / "data.yaml").is_file(), f"데이터셋 경로 확인: {DATASET}"

    # =========================================================
    # 2. Dataset YAML 생성 및 검증
    # =========================================================
    original = yaml.safe_load((DATASET / "data.yaml").read_text(encoding="utf-8"))
    source_names = original["names"]
    source_names = (
        [source_names[i] for i in range(len(source_names))]
        if isinstance(source_names, dict)
        else source_names
    )
    assert source_names == NAMES, f"클래스 순서 불일치: {source_names}"

    config = {
        "path": DATASET.as_posix(),
        "nc": len(NAMES),
        "names": dict(enumerate(NAMES)),
    }
    SPLITS = [s for s in ("train", "val", "test") if s in original]
    for split in SPLITS:
        assert (DATASET / "images" / split).is_dir(), (
            f"이미지 폴더 없음: {split}"
        )
        config[split] = f"images/{split}"

    DATA_YAML = OUTPUT / "data.local.yaml"
    DATA_YAML.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images_by_split = {}
    report, errors = {}, []

    for split in SPLITS:
        images = sorted(
            p
            for p in (DATASET / "images" / split).iterdir()
            if p.suffix.lower() in IMAGE_EXTS
        )
        images_by_split[split] = images
        counts = Counter()

        if not images:
            errors.append(f"{split}: 이미지 없음")
        stems = [p.stem for p in images]
        if len(stems) != len(set(stems)):
            errors.append(f"{split}: 중복 파일 stem")

        for p in images:
            label = DATASET / "labels" / split / (p.stem + ".txt")
            if not label.is_file():
                errors.append(f"라벨 누락: {label}")
                continue

            for line_no, line in enumerate(
                label.read_text(encoding="utf-8-sig").splitlines(), 1
            ):
                if not line.strip():
                    continue
                try:
                    values = list(map(float, line.split()))
                    assert len(values) == 5 and all(
                        math.isfinite(v) for v in values
                    )
                    cls, x, y, w, h = values
                    assert cls.is_integer() and 0 <= cls < len(NAMES)
                    assert 0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1
                    assert (
                        x - w / 2 >= -1e-5
                        and y - h / 2 >= -1e-5
                        and x + w / 2 <= 1 + 1e-5
                        and y + h / 2 <= 1 + 1e-5
                    )
                    counts[int(cls)] += 1
                except (ValueError, AssertionError):
                    errors.append(f"잘못된 라벨: {label}:{line_no}")

        for p in random.Random(42).sample(images, min(20, len(images))):
            try:
                with Image.open(p) as im:
                    im.verify()
            except Exception as exc:
                errors.append(f"이미지 읽기 실패: {p}: {exc}")

        report[split] = {
            "images": len(images),
            "boxes": {name: counts[i] for i, name in enumerate(NAMES)},
        }
        print(split, report[split])

        for i, name in enumerate(NAMES):
            if counts[i] == 0:
                print(f"주의: {split}에 {name} 객체가 없습니다.")

    (OUTPUT / "data_check.json").write_text(
        json.dumps({"splits": report, "errors": errors}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    assert not errors, f"검사 오류 {len(errors)}건. 예: {errors[:5]}"
    print("데이터셋 검사 통과")

    # 샘플 바운딩 박스 시각화
    samples = random.Random(42).sample(
        images_by_split["train"], min(4, len(images_by_split["train"]))
    )
    fig, axes = plt.subplots(
        1, len(samples), figsize=(5 * len(samples), 5), squeeze=False
    )
    for ax, p in zip(axes.flat, samples):
        with Image.open(p) as source:
            im = source.convert("RGB")
        width, height = im.size
        ax.imshow(im)
        for line in (
            (DATASET / "labels/train" / (p.stem + ".txt"))
            .read_text(encoding="utf-8-sig")
            .splitlines()
        ):
            if not line.strip():
                continue
            cls, x, y, w, h = map(float, line.split())
            left, top = (x - w / 2) * width, (y - h / 2) * height
            ax.add_patch(
                Rectangle(
                    (left, top),
                    w * width,
                    h * height,
                    fill=False,
                    edgecolor="lime",
                    linewidth=2,
                )
            )
            ax.text(
                left,
                top,
                NAMES[int(cls)],
                color="black",
                backgroundcolor="lime",
            )
        ax.set_title(p.name)
        ax.axis("off")
    plt.tight_layout()
    plt.show()

    # =========================================================
    # 3. 모델 학습 (GPU 적용)
    # =========================================================
    model = YOLO(MODEL_NAME)
    train_results = model.train(
        data=str(DATA_YAML),
        epochs=1 if SMOKE_TEST else EPOCHS,
        fraction=0.01 if SMOKE_TEST else 1.0,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        workers=WORKERS,
        project=str(OUTPUT / "runs"),
        name="smoke" if SMOKE_TEST else "train",
        exist_ok=True,
        patience=10,
        seed=42,
        deterministic=True,
        cache=False,
        plots=True,
    )

    RUN_DIR = Path(model.trainer.save_dir)
    BEST_WEIGHTS = RUN_DIR / "weights" / "best.pt"
    assert BEST_WEIGHTS.is_file(), f"학습 결과 확인: {RUN_DIR}"

    if not SMOKE_TEST:
        (OUTPUT / "latest_best.txt").write_text(
            str(BEST_WEIGHTS.resolve()), encoding="utf-8"
        )

    print("학습 완료 경로:", RUN_DIR)
    print("최적 가중치 경로:", BEST_WEIGHTS)

    # =========================================================
    # 4. 검증 및 추론 (GPU 적용)
    # =========================================================
    trained = YOLO(str(BEST_WEIGHTS))

    for split in [s for s in ("val", "test") if s in SPLITS]:
        metrics = trained.val(
            data=str(DATA_YAML),
            split=split,
            imgsz=IMGSZ,
            batch=BATCH,
            device=DEVICE,
            workers=WORKERS,
            project=str(OUTPUT / "evaluation"),
            name=split,
            exist_ok=True,
        )
        print(f"{split} mAP50: {metrics.box.map50}, mAP50-95: {metrics.box.map}")

    curve = RUN_DIR / "results.png"
    if curve.is_file():
        display(Image.open(curve))

    # 단일 이미지 테스트
    sample_path = images_by_split["val"][0]
    prediction = trained.predict(
        source=str(sample_path),
        conf=0.25,
        imgsz=IMGSZ,
        device=DEVICE,
        verbose=False,
    )[0]

    plt.figure(figsize=(9, 7))
    plt.imshow(cv2.cvtColor(prediction.plot(), cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

    # =========================================================
    # 5. 실시간 웹캠 추론 (GPU 적용)
    # =========================================================
    CAMERA_INDEX = 1
    CONFIDENCE = 0.35
    MAX_SECONDS = 300
    WEIGHTS_OVERRIDE = None

    if WEIGHTS_OVERRIDE:
        camera_weights = Path(WEIGHTS_OVERRIDE)
    else:
        pointer = OUTPUT / "latest_best.txt"
        if not pointer.is_file():
            raise FileNotFoundError(
                "전체 학습을 먼저 실행하거나 WEIGHTS_OVERRIDE에 학습된 best.pt 경로를 입력하세요."
            )
        camera_weights = Path(pointer.read_text(encoding="utf-8").strip())

    if not camera_weights.is_file():
        raise FileNotFoundError(camera_weights)

    camera_model = YOLO(str(camera_weights))
    cap = None
    window = "YOLO11 Recycling - Q or ESC to quit"

    try:
        cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(CAMERA_INDEX)
        if not cap.isOpened():
            raise RuntimeError(
                "카메라를 열지 못했습니다. 장치 번호와 Windows 카메라 권한을 확인하세요."
            )

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        started = time.perf_counter()

        while MAX_SECONDS is None or time.perf_counter() - started < MAX_SECONDS:
            frame_start = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                print("카메라 프레임을 읽지 못해 종료합니다.")
                break

            result = camera_model.predict(
                frame, conf=CONFIDENCE, imgsz=IMGSZ, device=DEVICE, verbose=False
            )[0]
            annotated = result.plot()

            fps = 1.0 / max(time.perf_counter() - frame_start, 1e-6)
            cv2.putText(
                annotated,
                f"FPS {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
            cv2.imshow(window, annotated)

            key = cv2.waitKey(1) & 0xFF
            if (
                key in (ord("q"), ord("Q"), 27)
                or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1
            ):
                break

    except KeyboardInterrupt:
        print("사용자가 카메라 인식을 중단했습니다.")
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()

    print("카메라 종료")
