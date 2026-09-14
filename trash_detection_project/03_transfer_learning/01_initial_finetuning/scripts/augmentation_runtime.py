"""픽셀만 바꾸는 증강. 이미지를 추가로 만들지 않고 목표 좌표도 바꾸지 않는다."""
import albumentations as A

def training_transforms():
    return [
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), sigma_limit=(0.2, 1.0), p=1),
            A.MotionBlur(blur_limit=(3, 5), allow_shifted=False, p=1),
        ], p=0.15),
        A.ImageCompression(quality_range=(75, 95), p=0.08),
    ]
