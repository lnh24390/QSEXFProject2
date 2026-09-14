from collections import Counter  # 클래스별 객체 수를 세기 위해 collections의 Counter를 가져옵니다.
import json  # 검사 결과를 JSON 형식으로 저장하기 위해 json 모듈을 가져옵니다.
import math  # 라벨 숫자가 유한한 값인지 확인하기 위해 math 모듈을 가져옵니다.
import os  # 실행 환경 변수를 설정하기 위해 os 모듈을 가져옵니다.
import random  # 검사·시각화용 이미지를 무작위로 선택하기 위해 random 모듈을 가져옵니다.
import time  # 웹캠 실행 시간과 프레임 처리 시간을 측정하기 위해 time 모듈을 가져옵니다.
from pathlib import Path  # 폴더·파일 경로를 구성하고 파일을 읽고 쓰기 위해 Path를 가져옵니다.

import cv2  # 웹캠 제어와 영상 처리를 위해 OpenCV를 cv2라는 이름으로 가져옵니다.
from IPython.display import display  # IPython 환경에서 이미지 객체를 표시하는 display 함수를 가져옵니다.
import matplotlib.pyplot as plt  # 이미지와 그래프를 표시하기 위해 pyplot을 plt라는 이름으로 가져옵니다.
from matplotlib.patches import Rectangle  # 이미지 위에 사각형 바운딩 박스를 그리기 위해 Rectangle을 가져옵니다.
from PIL import Image  # 이미지 열기, 형식 변환, 파일 검사를 위해 Pillow의 Image를 가져옵니다.
import torch  # CUDA 사용 가능 여부와 PyTorch 설정을 확인하기 위해 torch를 가져옵니다.
import yaml  # 데이터셋 설정 파일인 YAML을 읽고 쓰기 위해 yaml을 가져옵니다.
from ultralytics import YOLO  # YOLO 모델의 학습·평가·추론을 수행하기 위해 YOLO 클래스를 가져옵니다.

if __name__ == '__main__':  # 직접 실행한 경우에만 본문을 수행하여, Windows 자식 프로세스에서 학습 코드가 재실행되는 것을 막습니다.
    # =========================================================
    # 1. GPU / PyTorch 실행 환경 및 경로 최적화
    # =========================================================
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # OpenMP 런타임의 중복 로딩을 허용하도록 환경 변수를 설정합니다.

    if torch.cuda.is_available():  # 현재 PyTorch 환경에서 CUDA GPU를 사용할 수 있는지 확인합니다.
        torch.backends.cudnn.benchmark = True  # cuDNN의 연산 알고리즘 탐색을 켭니다. 입력 크기가 일정할 때 유리할 수 있으며 학습기가 재설정할 수도 있습니다.
        DEVICE = 0  # 연산에 사용할 첫 번째 CUDA GPU의 장치 번호를 지정합니다. 번호는 0부터 시작합니다.
        GPU_NAME = torch.cuda.get_device_name(0)  # 0번 GPU의 제품 이름을 문자열로 가져옵니다.
        WORKERS = 4  # 데이터 로딩에 사용할 작업 프로세스 수를 4개로 지정합니다.
        print(f"GPU 연결 성공: {GPU_NAME} (Device: {DEVICE})")  # 선택한 GPU의 이름과 장치 번호를 출력합니다. f 문자열의 중괄호에 변수 값을 넣습니다.
    else:  # CUDA GPU를 사용할 수 없을 때 아래 CPU 설정을 적용합니다.
        DEVICE = "cpu"  # 연산 장치를 CPU로 지정합니다.
        WORKERS = 0  # 별도 데이터 로더 프로세스를 만들지 않고 현재 프로세스에서 데이터를 읽도록 설정합니다.
        print("경고: CUDA 가능한 GPU를 찾을 수 없어 CPU 모드로 동작합니다.")  # CUDA GPU를 사용할 수 없어 CPU로 실행한다는 안내를 출력합니다.

    print("PyTorch 버전:", torch.__version__)  # 현재 실행 환경에 설치된 PyTorch 버전을 출력합니다.

    PROJECT = Path(r"D:/QSEXFProject2")  # 프로젝트의 기준 폴더를 Path 객체로 지정합니다. r 접두사는 역슬래시 이스케이프 해석을 막습니다.
    DATASET_NAME = "YOLO_7CLASS_10000_PER_CLASS_20260909"  # 사용할 데이터셋 폴더 이름을 문자열로 지정합니다.
    DATASET = PROJECT / DATASET_NAME  # Path의 / 연산자로 프로젝트 경로와 데이터셋 폴더 이름을 연결합니다.
    MODEL_NAME = "yolo11n.pt"  # 학습을 시작할 YOLO11 nano 모델의 가중치 파일 이름을 지정합니다.

    EPOCHS = 50  # 전체 학습 데이터를 반복해서 학습할 최대 횟수를 50 epoch로 지정합니다.
    IMGSZ = 640  # 모델 입력 이미지 크기의 기준값을 640으로 지정합니다.
    BATCH = 8  # 한 번의 학습 배치에 포함할 이미지 수를 8장으로 지정합니다.
    SMOKE_TEST = False  # 간단 학습 점검 모드를 끕니다. True이면 아래 train()에서 학습 데이터 1%로 1 epoch만 실행합니다.

    NAMES = ["paper", "plastic", "can", "glass", "battery", "vinyl", "general"]  # 클래스 ID 0~6을 종이·플라스틱·캔·유리·배터리·비닐·일반쓰레기 순서로 대응시킵니다.
    OUTPUT = PROJECT / "yolo11_output" / DATASET_NAME  # 데이터셋별 검사·학습·평가 결과를 저장할 출력 경로를 구성합니다.
    OUTPUT.mkdir(parents=True, exist_ok=True)  # 필요한 상위 폴더까지 생성하며, 폴더가 이미 존재해도 오류를 내지 않습니다.

    assert (DATASET / "data.yaml").is_file(), f"데이터셋 경로 확인: {DATASET}"  # 원본 data.yaml이 파일로 존재하는지 검사하고, 없으면 경로를 포함한 AssertionError를 발생시킵니다.

    # =========================================================
    # 2. Dataset YAML 생성 및 검증
    # =========================================================
    original = yaml.safe_load((DATASET / "data.yaml").read_text(encoding="utf-8"))  # data.yaml을 UTF-8로 읽은 뒤 safe_load로 YAML 내용을 파이썬 자료구조로 변환합니다.
    source_names = original["names"]  # 원본 설정에서 클래스 이름 목록 또는 클래스 ID·이름 딕셔너리를 꺼냅니다.
    source_names = (  # 여러 줄로 작성한 조건부 표현식의 결과를 source_names에 다시 저장합니다.
        [source_names[i] for i in range(len(source_names))]  # 딕셔너리이면 ID 순서대로 이름을 모읍니다. 키가 0부터 시작하는 연속 정수라고 가정합니다.
        if isinstance(source_names, dict)  # source_names가 딕셔너리일 때 바로 앞의 리스트 변환식을 선택합니다.
        else source_names  # 딕셔너리가 아니면 기존 source_names 값을 그대로 사용합니다.
    )  # 클래스 이름 형식을 통일하는 조건부 표현식의 괄호를 닫습니다.
    assert source_names == NAMES, f"클래스 순서 불일치: {source_names}"  # 클래스 이름과 순서가 NAMES와 정확히 일치하는지 검사합니다.

    config = {  # 현재 PC의 데이터셋 위치를 반영할 설정 딕셔너리를 만듭니다.
        "path": DATASET.as_posix(),  # 데이터셋 루트 경로를 / 구분자를 사용하는 문자열로 기록합니다.
        "nc": len(NAMES),  # 클래스 개수를 기록합니다. 현재 NAMES에는 7개 클래스가 있습니다.
        "names": dict(enumerate(NAMES)),  # enumerate로 0부터 ID를 붙인 뒤 {ID: 클래스 이름} 딕셔너리를 만듭니다.
    }  # 설정 딕셔너리 정의를 마칩니다.
    SPLITS = [s for s in ("train", "val", "test") if s in original]  # 원본 YAML에 등록된 train(학습), val(검증), test(테스트) 키만 순서대로 선택합니다.
    for split in SPLITS:  # 선택된 각 데이터 분할에 대해 이미지 경로를 설정합니다.
        assert (DATASET / "images" / split).is_dir(), (  # 현재 분할의 images/분할명 폴더가 있는지 검사하고, 실패할 때 사용할 메시지를 이어 씁니다.
            f"이미지 폴더 없음: {split}"  # 폴더가 없을 때 어떤 분할에서 문제가 생겼는지 알려 주는 오류 메시지입니다.
        )  # assert의 오류 메시지를 묶은 괄호를 닫습니다.
        config[split] = f"images/{split}"  # 원본 경로 값 대신 데이터셋 루트 기준의 images/분할명 경로를 설정합니다.

    DATA_YAML = OUTPUT / "data.local.yaml"  # 현재 PC용 설정 파일을 출력 폴더의 data.local.yaml로 지정합니다.
    DATA_YAML.write_text(  # 새로 구성한 데이터셋 설정을 파일에 기록합니다.
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False),  # 설정을 YAML 문자열로 변환하며, 한글을 그대로 표시하고 항목의 삽입 순서를 유지합니다.
        encoding="utf-8",  # 설정 파일을 UTF-8 인코딩으로 저장합니다.
    )  # 설정 파일을 저장하는 write_text 호출을 마칩니다.

    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}  # 이미지 경로를 선별할 때 허용할 확장자를 집합으로 정의합니다.
    images_by_split = {}  # 분할별 이미지 경로 목록을 보관할 빈 딕셔너리를 만듭니다.
    report, errors = {}, []  # 검사 통계를 담을 딕셔너리와 오류를 누적할 리스트를 각각 초기화합니다.

    for split in SPLITS:  # 각 데이터 분할의 이미지와 라벨을 검사합니다.
        images = sorted(  # 조건을 만족하는 이미지 경로를 정렬하여 리스트로 만듭니다.
            p  # 아래 반복·필터 조건을 통과한 경로 p 자체를 정렬 대상으로 전달합니다.
            for p in (DATASET / "images" / split).iterdir()  # 분할 폴더 바로 아래의 경로를 하나씩 가져옵니다. 하위 폴더까지 재귀 탐색하지 않습니다.
            if p.suffix.lower() in IMAGE_EXTS  # 확장자를 소문자로 바꾼 뒤 허용된 이미지 확장자에 해당하는 경로만 남깁니다.
        )  # 이미지 경로를 정렬하는 sorted 호출을 마칩니다.
        images_by_split[split] = images  # 현재 분할의 이미지 목록을 이후 시각화와 추론에서도 사용할 수 있도록 보관합니다.
        counts = Counter()  # 현재 분할의 클래스별 바운딩 박스 수를 셀 Counter를 초기화합니다.

        if not images:  # 선택된 이미지가 한 장도 없는지 확인합니다.
            errors.append(f"{split}: 이미지 없음")  # 현재 분할에 이미지가 없다는 내용을 오류 목록에 추가합니다.
        stems = [p.stem for p in images]  # 각 이미지의 확장자를 제외한 파일명인 stem을 모읍니다.
        if len(stems) != len(set(stems)):  # 중복 제거 전후의 개수가 다르면 같은 stem이 여러 번 등장한 것입니다.
            errors.append(f"{split}: 중복 파일 stem")  # a.jpg와 a.png처럼 같은 라벨 파일을 참조할 수 있는 파일명 중복을 기록합니다.

        for p in images:  # 현재 분할의 이미지 경로를 하나씩 검사합니다.
            label = DATASET / "labels" / split / (p.stem + ".txt")  # 이미지와 stem이 같은 labels/분할명/파일명.txt 경로를 구성합니다.
            if not label.is_file():  # 이미지에 대응하는 라벨 파일이 없는지 확인합니다.
                errors.append(f"라벨 누락: {label}")  # 누락된 라벨의 경로를 오류 목록에 추가합니다.
                continue  # 현재 이미지의 나머지 검사를 건너뛰고 다음 이미지로 넘어갑니다.

            for line_no, line in enumerate(  # 라벨 파일을 한 줄씩 순회하면서 line_no에는 줄 번호, line에는 줄 내용을 저장합니다.
                label.read_text(encoding="utf-8-sig").splitlines(), 1  # UTF-8 BOM이 있으면 제거해 읽고 줄 단위로 나눕니다. enumerate의 시작 번호는 1입니다.
            ):  # enumerate 호출을 마치고 각 라벨 줄을 처리하는 반복문 본문을 시작합니다.
                if not line.strip():  # 앞뒤 공백을 제거했을 때 내용이 없는 빈 줄인지 확인합니다.
                    continue  # 빈 줄을 건너뜁니다. 따라서 빈 라벨 파일 자체는 이 검사에서 오류가 아닙니다.
                try:  # 숫자 변환 실패와 라벨 규칙 위반을 처리할 수 있도록 검사 구간을 시작합니다.
                    values = list(map(float, line.split()))  # 공백으로 구분된 각 값을 float로 변환하여 리스트로 만듭니다.
                    assert len(values) == 5 and all(  # 라벨 값이 정확히 5개인지 검사하고, 모든 값의 유한성도 확인합니다.
                        math.isfinite(v) for v in values  # 각 값이 NaN이나 무한대가 아닌지 검사한 결과를 all에 전달합니다.
                    )  # 모든 라벨 값의 유한성을 검사하는 all 호출을 마칩니다.
                    cls, x, y, w, h = values  # 클래스 ID, 정규화된 중심 x·y, 정규화된 너비·높이를 각각 꺼냅니다.
                    assert cls.is_integer() and 0 <= cls < len(NAMES)  # 클래스 ID가 정수 값을 가지며 0 이상 클래스 개수 미만인지 검사합니다.
                    assert 0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1  # 중심 좌표는 0~1, 너비와 높이는 0 초과 1 이하인지 검사합니다.
                    assert (  # 바운딩 박스의 네 변이 이미지 범위를 벗어나지 않는지 확인하는 조건을 묶습니다.
                        x - w / 2 >= -1e-5  # 박스의 왼쪽 변을 검사합니다. 1e-5의 작은 좌표 오차는 허용합니다.
                        and y - h / 2 >= -1e-5  # 왼쪽 변 조건과 함께 박스의 위쪽 변도 이미지 범위 안인지 검사합니다.
                        and x + w / 2 <= 1 + 1e-5  # 박스의 오른쪽 변이 정규화된 이미지 경계 1을 허용 오차 이상 넘지 않는지 검사합니다.
                        and y + h / 2 <= 1 + 1e-5  # 박스의 아래쪽 변이 정규화된 이미지 경계 1을 허용 오차 이상 넘지 않는지 검사합니다.
                    )  # 네 변에 대한 모든 조건을 묶은 assert 표현식을 마칩니다.
                    counts[int(cls)] += 1  # 검사를 통과한 객체의 클래스 ID를 정수로 변환하고 해당 클래스의 객체 수를 1 늘립니다.
                except (ValueError, AssertionError):  # float 변환 실패 또는 assert 검사 실패를 잡아 전체 검사를 계속합니다.
                    errors.append(f"잘못된 라벨: {label}:{line_no}")  # 잘못된 라벨 파일 경로와 줄 번호를 오류 목록에 기록합니다.

        for p in random.Random(42).sample(images, min(20, len(images))):  # 고정 시드 42로 최대 20개 이미지를 중복 없이 선택하여 표본 검사합니다.
            try:  # 이미지 파일을 열거나 검사하는 과정에서 생길 예외를 처리합니다.
                with Image.open(p) as im:  # Pillow로 이미지를 열고, with 구문을 벗어나면 파일을 자동으로 닫습니다.
                    im.verify()  # Pillow가 검사할 수 있는 이미지 파일의 무결성 오류를 확인합니다.
            except Exception as exc:  # 이미지 검사 중 발생한 예외 객체를 exc에 저장합니다.
                errors.append(f"이미지 읽기 실패: {p}: {exc}")  # 읽기 또는 검사에 실패한 이미지 경로와 예외 내용을 기록합니다.

        report[split] = {  # 현재 분할의 검사 통계를 딕셔너리로 구성합니다.
            "images": len(images),  # 현재 분할에서 찾은 이미지 수를 기록합니다.
            "boxes": {name: counts[i] for i, name in enumerate(NAMES)},  # 클래스별 바운딩 박스 수를 이름에 대응시킵니다. 이미지 수가 아닌 객체 수입니다.
        }  # 현재 분할의 통계 딕셔너리 정의를 마칩니다.
        print(split, report[split])  # 현재 분할 이름과 이미지·객체 수 통계를 출력합니다.

        for i, name in enumerate(NAMES):  # 각 클래스의 ID i와 이름 name을 함께 순회합니다.
            if counts[i] == 0:  # 현재 분할에서 해당 클래스의 유효한 객체가 하나도 없는지 확인합니다.
                print(f"주의: {split}에 {name} 객체가 없습니다.")  # 객체가 없는 클래스를 안내합니다. 이 경우는 errors 목록에는 추가하지 않습니다.

    (OUTPUT / "data_check.json").write_text(  # 출력 폴더의 data_check.json에 검사 보고서를 기록합니다.
        json.dumps({"splits": report, "errors": errors}, ensure_ascii=False, indent=2),  # 통계와 오류를 JSON 문자열로 만들며, 한글을 유지하고 2칸 들여쓰기를 적용합니다.
        encoding="utf-8",  # 검사 보고서를 UTF-8 인코딩으로 저장합니다.
    )  # 검사 보고서를 저장하는 write_text 호출을 마칩니다.
    assert not errors, f"검사 오류 {len(errors)}건. 예: {errors[:5]}"  # 오류가 있으면 개수와 앞의 최대 5건을 표시하는 AssertionError로 중단합니다.
    print("데이터셋 검사 통과")  # 누적된 검사 오류가 없으면 통과 메시지를 출력합니다.

    # 샘플 바운딩 박스 시각화
    samples = random.Random(42).sample(  # 고정 시드 42를 사용하는 별도 난수 생성기로 시각화할 이미지를 선택합니다.
        images_by_split["train"], min(4, len(images_by_split["train"]))  # 학습 이미지 목록에서 최대 4장을 중복 없이 고릅니다. train 분할이 필요합니다.
    )  # 샘플 이미지를 선택하는 sample 호출을 마칩니다.
    fig, axes = plt.subplots(  # 전체 그림 객체 fig와 각 이미지 표시 영역인 axes를 만듭니다.
        1, len(samples), figsize=(5 * len(samples), 5), squeeze=False  # 1행으로 배치하고 이미지마다 너비 5인치를 할당합니다. squeeze=False로 axes를 2차원으로 유지합니다.
    )  # 시각화 영역을 생성하는 subplots 호출을 마칩니다.
    for ax, p in zip(axes.flat, samples):  # axes.flat으로 표시 영역을 순회하며, zip으로 각 영역과 샘플 이미지 경로를 짝지어 처리합니다.
        with Image.open(p) as source:  # 현재 샘플 이미지 파일을 열고 with 구문이 끝나면 파일을 닫습니다.
            im = source.convert("RGB")  # 이미지를 RGB 3채널로 변환하여 화면에 표시할 객체를 만듭니다.
        width, height = im.size  # 원본 이미지의 픽셀 너비와 높이를 꺼냅니다.
        ax.imshow(im)  # 현재 표시 영역에 원본 이미지를 그립니다.
        for line in (  # 현재 이미지에 대응하는 학습 라벨을 줄 단위로 순회합니다.
            (DATASET / "labels/train" / (p.stem + ".txt"))  # labels/train 폴더에서 이미지와 stem이 같은 .txt 라벨 경로를 구성합니다.
            .read_text(encoding="utf-8-sig")  # 라벨 내용을 UTF-8로 읽으며, BOM이 있으면 제거합니다.
            .splitlines()  # 읽어 온 문자열을 줄별 문자열 목록으로 나눕니다.
        ):  # 라벨 읽기 표현식을 마치고 각 줄을 처리하는 반복문 본문을 시작합니다.
            if not line.strip():  # 공백만 있거나 내용이 없는 줄인지 확인합니다.
                continue  # 빈 줄은 시각화 대상에서 제외하고 다음 줄로 넘어갑니다.
            cls, x, y, w, h = map(float, line.split())  # 라벨의 다섯 값을 실수로 읽어 클래스 ID와 정규화된 중심·크기로 나눕니다.
            left, top = (x - w / 2) * width, (y - h / 2) * height  # 중심에서 크기의 절반을 빼고 원본 크기를 곱해 박스 왼쪽 위의 픽셀 좌표를 구합니다.
            ax.add_patch(  # 현재 이미지 표시 영역에 사각형 도형을 추가합니다.
                Rectangle(  # 바운딩 박스를 나타낼 Rectangle 객체를 생성합니다.
                    (left, top),  # 사각형의 시작점에 해당하는 왼쪽 위 모서리 좌표를 지정합니다.
                    w * width,  # 정규화된 박스 너비에 이미지 너비를 곱해 픽셀 너비를 구합니다.
                    h * height,  # 정규화된 박스 높이에 이미지 높이를 곱해 픽셀 높이를 구합니다.
                    fill=False,  # 사각형 내부를 채우지 않고 테두리만 표시합니다.
                    edgecolor="lime",  # 바운딩 박스 테두리 색을 라임색으로 지정합니다.
                    linewidth=2,  # 테두리 선의 두께를 2로 지정합니다.
                )  # Rectangle 생성을 마치고 도형 객체를 add_patch에 전달합니다.
            )  # 현재 이미지 위에 바운딩 박스를 추가하는 호출을 마칩니다.
            ax.text(  # 바운딩 박스 근처에 클래스 이름을 글자로 표시합니다.
                left,  # 글자를 표시할 가로 좌표를 박스의 왼쪽 좌표로 지정합니다.
                top,  # 글자를 표시할 세로 좌표를 박스의 위쪽 좌표로 지정합니다.
                NAMES[int(cls)],  # 실수로 읽은 클래스 ID를 정수 인덱스로 바꾸어 해당 클래스 이름을 가져옵니다.
                color="black",  # 글자 색을 검정색으로 지정합니다.
                backgroundcolor="lime",  # 글자 배경을 라임색으로 지정하여 읽기 쉽게 만듭니다.
            )  # 클래스 이름을 표시하는 text 호출을 마칩니다.
        ax.set_title(p.name)  # 각 이미지 표시 영역의 제목에 파일명을 표시합니다.
        ax.axis("off")  # 현재 이미지 표시 영역의 좌표축과 눈금을 숨깁니다.
    plt.tight_layout()  # 제목과 이미지가 덜 겹치도록 표시 영역 사이 간격을 자동 조정합니다.
    plt.show()  # 샘플 이미지와 정답 바운딩 박스를 표시합니다.

    # =========================================================
    # 3. 모델 학습 (GPU 적용)
    # =========================================================
    model = YOLO(MODEL_NAME)  # 지정한 YOLO11 가중치를 불러와 학습할 모델 객체를 만듭니다.
    train_results = model.train(  # 모델 학습을 실행하고 반환 결과를 train_results에 저장합니다.
        data=str(DATA_YAML),  # 앞에서 생성한 로컬 데이터셋 YAML 경로를 문자열로 전달합니다.
        epochs=1 if SMOKE_TEST else EPOCHS,  # 간단 점검 모드이면 1 epoch, 아니면 EPOCHS에 지정한 최대 횟수만큼 학습합니다.
        fraction=0.01 if SMOKE_TEST else 1.0,  # 간단 점검 모드이면 학습 데이터 1%, 아니면 전체를 사용합니다.
        imgsz=IMGSZ,  # 학습용 입력 이미지 크기의 기준값을 전달합니다.
        batch=BATCH,  # 한 배치에서 처리할 이미지 수를 전달합니다.
        device=DEVICE,  # 학습에 사용할 GPU 번호 또는 CPU 지정을 전달합니다.
        workers=WORKERS,  # 데이터 로더에 사용할 작업 프로세스 수를 전달합니다.
        project=str(OUTPUT / "runs"),  # 학습 실행 결과를 모을 상위 폴더를 지정합니다.
        name="smoke" if SMOKE_TEST else "train",  # 실행 결과 하위 폴더 이름을 점검 모드에서는 smoke, 일반 학습에서는 train으로 지정합니다.
        exist_ok=True,  # 같은 이름의 결과 폴더가 이미 있으면 재사용하도록 허용합니다.
        patience=10,  # 검증 성능 개선이 없는 상태가 10 epoch 이어지면 조기 종료하도록 설정합니다.
        seed=42,  # 학습에 사용할 난수 시드를 42로 지정합니다.
        deterministic=True,  # 학습 재현성을 높이기 위해 결정적 연산 사용을 요청합니다.
        cache=False,  # 데이터셋 이미지 캐싱을 사용하지 않도록 설정합니다.
        plots=True,  # 학습 곡선 등 결과 그래프를 생성하도록 설정합니다.
    )  # 학습 설정 인자 전달을 마치고 train 호출의 결과를 받습니다.

    RUN_DIR = Path(model.trainer.save_dir)  # 학습기가 실제로 사용한 결과 저장 폴더를 Path 객체로 가져옵니다.
    BEST_WEIGHTS = RUN_DIR / "weights" / "best.pt"  # 검증 성능을 기준으로 선택된 최적 가중치 best.pt의 경로를 구성합니다.
    assert BEST_WEIGHTS.is_file(), f"학습 결과 확인: {RUN_DIR}"  # best.pt가 실제 파일로 존재하는지 확인하고, 없으면 학습 결과 폴더를 안내하며 중단합니다.

    if not SMOKE_TEST:  # 간단 점검 모드가 아닌 일반 학습인 경우에만 아래 경로 기록을 갱신합니다.
        (OUTPUT / "latest_best.txt").write_text(  # 웹캠 단계에서 읽을 최적 가중치 경로 기록 파일을 작성합니다.
            str(BEST_WEIGHTS.resolve()), encoding="utf-8"  # best.pt 경로를 절대 경로로 해석한 뒤 문자열로 바꾸어 UTF-8로 저장합니다.
        )  # 최적 가중치 경로를 기록하는 write_text 호출을 마칩니다.

    print("학습 완료 경로:", RUN_DIR)  # 학습 결과가 저장된 폴더 경로를 출력합니다.
    print("최적 가중치 경로:", BEST_WEIGHTS)  # 이번 학습의 최적 가중치 파일 경로를 출력합니다.

    # =========================================================
    # 4. 검증 및 추론 (GPU 적용)
    # =========================================================
    trained = YOLO(str(BEST_WEIGHTS))  # 이번 학습에서 생성한 best.pt로 평가·이미지 추론용 모델을 불러옵니다.

    for split in [s for s in ("val", "test") if s in SPLITS]:  # 등록된 분할 중 val과 test만 골라 각각 평가합니다.
        metrics = trained.val(  # 선택한 분할에 대한 모델 평가를 실행하고 지표 객체를 metrics에 저장합니다.
            data=str(DATA_YAML),  # 평가할 데이터셋의 설정 파일 경로를 전달합니다.
            split=split,  # 이번에 평가할 분할 이름인 val 또는 test를 지정합니다.
            imgsz=IMGSZ,  # 평가용 입력 이미지 크기의 기준값을 전달합니다.
            batch=BATCH,  # 평가 시 한 배치에서 처리할 이미지 수를 전달합니다.
            device=DEVICE,  # 평가에 사용할 GPU 또는 CPU를 지정합니다.
            workers=WORKERS,  # 평가 데이터를 읽을 데이터 로더 작업 프로세스 수를 전달합니다.
            project=str(OUTPUT / "evaluation"),  # 평가 결과를 저장할 상위 폴더를 지정합니다.
            name=split,  # 현재 분할 이름으로 평가 결과 하위 폴더를 지정합니다.
            exist_ok=True,  # 같은 이름의 평가 결과 폴더가 이미 있으면 재사용합니다.
        )  # 평가 설정 인자 전달을 마치고 val 호출의 결과를 받습니다.
        print(f"{split} mAP50: {metrics.box.map50}, mAP50-95: {metrics.box.map}")  # IoU 0.50 기준 mAP와 IoU 0.50~0.95를 0.05 간격으로 평가한 평균 mAP를 출력합니다.

    curve = RUN_DIR / "results.png"  # 학습 중 생성된 지표·손실 그래프 이미지의 경로를 구성합니다.
    if curve.is_file():  # 학습 결과 그래프 파일이 실제로 존재하는지 확인합니다.
        display(Image.open(curve))  # 그래프 이미지를 열어 IPython의 이미지 표시 기능에 전달합니다.

    # 단일 이미지 테스트
    sample_path = images_by_split["val"][0]  # 검증 이미지 목록의 첫 번째 경로를 선택합니다. val 분할에 이미지가 있어야 합니다.
    prediction = trained.predict(  # 학습한 모델로 단일 이미지의 객체 탐지를 실행합니다.
        source=str(sample_path),  # 추론할 이미지 파일 경로를 문자열로 전달합니다.
        conf=0.25,  # 탐지 결과를 남길 최소 신뢰도 임계값을 0.25로 설정합니다.
        imgsz=IMGSZ,  # 추론용 입력 이미지 크기의 기준값을 전달합니다.
        device=DEVICE,  # 추론에 사용할 GPU 또는 CPU를 지정합니다.
        verbose=False,  # 이미지별 상세 추론 로그 출력을 줄입니다.
    )[0]  # predict 호출을 마치고 반환된 결과 목록에서 첫 번째 이미지 결과를 꺼냅니다.

    plt.figure(figsize=(9, 7))  # 추론 결과를 표시할 그림 크기를 가로 9인치, 세로 7인치로 지정합니다.
    plt.imshow(cv2.cvtColor(prediction.plot(), cv2.COLOR_BGR2RGB))  # 탐지 결과를 그린 BGR 이미지를 RGB로 변환하여 Matplotlib에 표시합니다.
    plt.axis("off")  # 추론 이미지 주변의 좌표축과 눈금을 숨깁니다.
    plt.show()  # 단일 이미지의 객체 탐지 결과를 표시합니다.

    # =========================================================
    # 5. 실시간 웹캠 추론 (GPU 적용)
    # =========================================================
    CAMERA_INDEX = 1  # 열 카메라의 장치 번호를 1로 지정합니다. 실제 장치 대응은 PC 구성에 따라 달라집니다.
    CONFIDENCE = 0.35  # 웹캠에서 탐지 결과를 남길 최소 신뢰도를 0.35로 지정합니다.
    MAX_SECONDS = 300  # 웹캠 반복문의 실행 시간을 300초로 제한합니다. None으로 지정하면 시간 제한이 없습니다.
    WEIGHTS_OVERRIDE = None  # 직접 사용할 가중치 경로의 초기값입니다. 경로를 지정하면 기록 파일보다 우선 사용합니다.

    if WEIGHTS_OVERRIDE:  # 직접 지정한 가중치 경로가 있는지 확인합니다.
        camera_weights = Path(WEIGHTS_OVERRIDE)  # 사용자가 지정한 가중치 경로를 Path 객체로 변환합니다.
    else:  # 직접 지정한 경로가 없으면 기록 파일에서 가중치 경로를 읽습니다.
        pointer = OUTPUT / "latest_best.txt"  # 일반 학습에서 기록한 최적 가중치 경로 파일을 지정합니다. 점검 학습은 이 파일을 갱신하지 않습니다.
        if not pointer.is_file():  # 최적 가중치 경로 기록 파일이 없는지 확인합니다.
            raise FileNotFoundError(  # 기록 파일이 없으면 FileNotFoundError를 발생시킵니다.
                "전체 학습을 먼저 실행하거나 WEIGHTS_OVERRIDE에 학습된 best.pt 경로를 입력하세요."  # 일반 학습을 실행하거나 가중치 경로를 직접 지정해야 한다는 오류 메시지입니다.
            )  # 오류 객체 생성을 마치고 예외를 발생시킵니다.
        camera_weights = Path(pointer.read_text(encoding="utf-8").strip())  # 기록된 경로의 앞뒤 공백과 줄바꿈을 제거한 뒤 가중치 경로 객체를 만듭니다.

    if not camera_weights.is_file():  # 선택한 경로에 실제 가중치 파일이 없는지 확인합니다.
        raise FileNotFoundError(camera_weights)  # 가중치 파일이 없으면 해당 경로를 포함한 FileNotFoundError로 중단합니다.

    camera_model = YOLO(str(camera_weights))  # 선택한 가중치를 불러와 웹캠 추론용 YOLO 모델을 만듭니다.
    cap = None  # 카메라 연결 전 상태로 초기화하여 finally에서 정리할 객체가 있는지 확인할 수 있게 합니다.
    window = "YOLO11 Recycling - Q or ESC to quit"  # 웹캠 영상 창의 이름과 종료 키 안내 문구를 지정합니다.

    try:  # 카메라 실행을 시작하며, 중단·오류 여부와 관계없이 finally에서 자원을 정리하도록 합니다.
        cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)  # Windows의 DirectShow 방식으로 지정한 번호의 카메라 연결을 시도합니다.
        if not cap.isOpened():  # 첫 번째 방식으로 카메라를 열지 못했는지 확인합니다.
            cap.release()  # 실패한 카메라 연결 객체의 자원을 해제합니다.
            cap = cv2.VideoCapture(CAMERA_INDEX)  # 백엔드를 직접 지정하지 않는 기본 방식으로 같은 번호의 카메라를 다시 엽니다.
        if not cap.isOpened():  # 재시도 후에도 카메라를 열지 못했는지 확인합니다.
            raise RuntimeError(  # 카메라를 사용할 수 없으면 RuntimeError를 발생시킵니다.
                "카메라를 열지 못했습니다. 장치 번호와 Windows 카메라 권한을 확인하세요."  # 카메라 장치 번호와 Windows 카메라 권한을 확인하도록 안내하는 오류 메시지입니다.
            )  # 오류 객체 생성을 마치고 예외를 발생시킵니다.

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # 카메라 입력 영상 너비를 640픽셀로 요청합니다. 실제 적용은 장치 지원에 따라 다릅니다.
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # 카메라 입력 영상 높이를 480픽셀로 요청합니다. 실제 적용은 장치 지원에 따라 다릅니다.
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)  # 사용자가 크기를 조절할 수 있는 영상 표시 창을 만듭니다.
        started = time.perf_counter()  # 전체 웹캠 실행 시간의 기준이 될 시작 시각을 고해상도 타이머로 기록합니다.

        while MAX_SECONDS is None or time.perf_counter() - started < MAX_SECONDS:  # 시간 제한이 없거나 경과 시간이 제한보다 짧으면 프레임 처리를 반복합니다.
            frame_start = time.perf_counter()  # 현재 프레임의 처리 시간을 계산하기 위한 시작 시각을 기록합니다.
            ok, frame = cap.read()  # 카메라에서 프레임을 읽습니다. ok는 성공 여부이고 frame은 BGR 영상 배열입니다.
            if not ok:  # 현재 프레임을 정상적으로 읽지 못했는지 확인합니다.
                print("카메라 프레임을 읽지 못해 종료합니다.")  # 프레임 읽기에 실패하여 웹캠 실행을 종료한다는 메시지를 출력합니다.
                break  # 웹캠 프레임 처리 반복문을 빠져나갑니다.

            result = camera_model.predict(  # 현재 카메라 프레임에서 객체 탐지를 실행합니다.
                frame, conf=CONFIDENCE, imgsz=IMGSZ, device=DEVICE, verbose=False  # 프레임, 최소 신뢰도, 입력 크기, 연산 장치를 전달하고 상세 추론 로그는 줄입니다.
            )[0]  # 반환된 결과 목록에서 현재 프레임에 해당하는 첫 번째 결과 객체를 선택합니다.
            annotated = result.plot()  # 탐지 박스·클래스 이름·신뢰도를 그린 BGR 영상을 만듭니다.

            fps = 1.0 / max(time.perf_counter() - frame_start, 1e-6)  # 읽기·추론·박스 그리기 시간의 역수로 FPS를 추정합니다. 1e-6으로 0 나눗셈을 막으며 아래 화면 출력 시간은 제외합니다.
            cv2.putText(  # 추론 결과 영상에 FPS 안내 글자를 추가합니다.
                annotated,  # 글자를 그릴 대상 영상 배열을 전달합니다.
                f"FPS {fps:.1f}",  # FPS 값을 소수점 첫째 자리까지 표시할 문자열을 만듭니다.
                (10, 30),  # 글자의 시작 위치를 영상의 (x=10, y=30) 픽셀로 지정합니다.
                cv2.FONT_HERSHEY_SIMPLEX,  # OpenCV에 내장된 Hershey Simplex 글꼴을 선택합니다.
                0.8,  # 글꼴 크기 배율을 0.8로 지정합니다.
                (0, 255, 0),  # BGR 순서의 색상 값 (0, 255, 0)으로 초록색 글자를 지정합니다.
                2,  # 글자 선 두께를 2로 지정합니다.
            )  # FPS 글자를 영상에 추가하는 putText 호출을 마칩니다.
            cv2.imshow(window, annotated)  # 객체 탐지 결과와 FPS가 그려진 영상을 지정한 창에 표시합니다.

            key = cv2.waitKey(1) & 0xFF  # 짧게 키 입력·창 이벤트를 처리하고, 입력 키 코드의 하위 8비트만 남깁니다.
            if (  # 키 입력이나 창 상태를 확인하여 반복문 종료 여부를 판단합니다.
                key in (ord("q"), ord("Q"), 27)  # q, Q 또는 ESC 키(코드 27)가 입력되었는지 확인합니다.
                or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1  # 또는 창의 표시 상태를 확인하여 창이 닫혔는지 판단합니다.
            ):  # 종료 조건을 묶은 괄호를 닫고, 조건이 참일 때 아래 본문을 실행합니다.
                break  # 종료 키를 누르거나 창을 닫으면 웹캠 반복문을 종료합니다.

    except KeyboardInterrupt:  # Ctrl+C 등으로 발생한 사용자의 실행 중단을 처리합니다.
        print("사용자가 카메라 인식을 중단했습니다.")  # 사용자가 카메라 인식을 중단했다는 메시지를 출력합니다.
    finally:  # 정상 종료·사용자 중단·예외 발생 모두에서 아래 자원 정리 코드를 실행합니다.
        if cap is not None:  # 카메라 객체가 만들어진 적이 있는지 확인합니다.
            cap.release()  # 카메라 연결과 관련 자원을 해제합니다.
        cv2.destroyAllWindows()  # OpenCV로 연 영상 창들을 닫습니다.

    print("카메라 종료")  # 정상적으로 이 줄에 도달하면 카메라 종료 메시지를 출력합니다.
