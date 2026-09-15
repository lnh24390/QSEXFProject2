import json
from pathlib import Path

DEFAULT_CONFIG = {
    "source": 0,
    "width": 1280,
    "height": 720,
    "conf_threshold": 0.5,
    "iou_threshold": 0.45,
    "imgsz": 640,
    "device": "auto",
    "quantize": None,  # None(FP32) | 16(FP16) | 8(INT8)
    "max_det": 300,
    "model_name": None,
    "region": "default",  # 분리수거 안내 지역: default(전국 공통) | jeju | seoul 등 (config/recycling_guide.json 참고)
    "guide_max_items": 3,
    "exposure": None,  # None=자동노출 | 숫자=수동(작을수록 어둡다). 역광일 때 수동으로 밝혀야 할 수 있다
}


def load_config(path):
    config = dict(DEFAULT_CONFIG)

    path = Path(path)
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as f:
                saved = json.load(f)
            # 예전 설정 파일의 half: true 를 quantize: 16 으로 옮긴다.
            if saved.pop("half", False) and "quantize" not in saved:
                saved["quantize"] = 16
            config.update({k: v for k, v in saved.items() if k in DEFAULT_CONFIG})
        except (json.JSONDecodeError, OSError) as exc:
            print(f">> 설정 파일을 읽지 못했습니다 ({exc}). 기본값을 사용합니다.")

    return config


def save_config(path, config):
    path = Path(path)
    try:
        with path.open("w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print(f">> 설정을 저장했습니다: {path}")
    except OSError as exc:
        print(f">> 설정 저장 실패: {exc}")


def reset_config(path):
    """config.json 을 기본값으로 되돌린다. 모델 선택 등 실행 중 저장된 값도 모두 지워진다."""
    config = dict(DEFAULT_CONFIG)
    save_config(path, config)
    return config
