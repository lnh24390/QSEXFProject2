import 'dart:ui';

import 'package:flutter_test/flutter_test.dart';
import 'package:trash_sorter/services/mask_image.dart';
import 'package:trash_sorter/services/photo_analysis.dart';
import 'package:ultralytics_yolo/ultralytics_yolo.dart';

const _red = 0xFFFF0000;
const _blue = 0xFF0000FF;

PhotoDetection _det(
  String name,
  double conf, {
  List<List<double>>? mask,
}) {
  final map = <String, dynamic>{
    'classIndex': 0,
    'className': name,
    'confidence': conf,
    'boundingBox': {'left': 0.0, 'top': 0.0, 'right': 10.0, 'bottom': 10.0},
    'normalizedBox': {'left': 0.0, 'top': 0.0, 'right': 1.0, 'bottom': 1.0},
    'mask': ?mask,
  };
  return PhotoDetection(result: YOLOResult.fromMap(map));
}

/// (x, y) 픽셀의 알파값.
int _alpha(MaskPixels p, int x, int y) => p.rgba[(y * p.width + x) * 4 + 3];

/// (x, y) 픽셀의 ARGB 값을 읽는다. 완전 투명이면 0.
int _pixel(MaskPixels p, int x, int y) {
  final i = (y * p.width + x) * 4;
  final a = p.rgba[i + 3];
  if (a == 0) return 0;
  return (a << 24) | (p.rgba[i] << 16) | (p.rgba[i + 1] << 8) | p.rgba[i + 2];
}

void main() {
  // 픽셀은 프리멀티플라이(색 × 알파)로 저장된다. 알파 255 인 외곽선 픽셀만 원색 그대로다.
  test('마스크가 없으면(detect 모델) null 을 반환한다', () {
    final pixels = buildMaskPixels(
      [_det('can', 0.9), _det('paper', 0.8)],
      colorOf: (_) => _red,
      threshold: 0.5,
      fillOpacity: 0.4,
      strokeWidth: 1,
    );
    expect(pixels, isNull);
  });

  test('기준 이상인 픽셀만 칠하고 나머지는 투명하다', () {
    final pixels = buildMaskPixels(
      [
        _det('can', 0.9, mask: [
          [0.9, 0.1],
          [0.5, 0.49],
        ]),
      ],
      colorOf: (_) => _red,
      threshold: 0.5,
      fillOpacity: 0.4,
      strokeWidth: 1,
    )!;
    expect(pixels.width, 2);
    expect(pixels.height, 2);
    expect(_pixel(pixels, 0, 0), _red); // 0.9 ≥ 0.5
    expect(_pixel(pixels, 1, 0), 0); // 0.1 미만
    expect(_pixel(pixels, 0, 1), _red); // 경계값 0.5 포함
    expect(_pixel(pixels, 1, 1), 0); // 0.49 미만
  });

  test('겹치는 부분은 신뢰도가 높은 검출 색이 이긴다', () {
    final pixels = buildMaskPixels(
      [
        _det('can', 0.6, mask: [
          [1.0, 1.0],
        ]),
        _det('paper', 0.9, mask: [
          [1.0, 0.0],
        ]),
      ],
      colorOf: (d) => d.className == 'paper' ? _blue : _red,
      threshold: 0.5,
      fillOpacity: 0.4,
      strokeWidth: 1,
    )!;
    expect(_pixel(pixels, 0, 0), _blue); // 겹침 → 신뢰도 0.9 가 이김
    expect(_pixel(pixels, 1, 0), _red); // can 만 있는 곳
  });

  test('안쪽은 옅게, 가장자리는 불투명하게 칠한다 (사각형 박스 대신 외곽선)', () {
    // 5x5 를 모두 채우면 바깥 테두리는 외곽선, 가운데는 채움이 된다
    final mask = List.generate(5, (_) => List.filled(5, 1.0));
    final pixels = buildMaskPixels(
      [_det('can', 0.9, mask: mask)],
      colorOf: (_) => _red,
      threshold: 0.5,
      fillOpacity: 0.4,
      strokeWidth: 1,
    )!;
    expect(_alpha(pixels, 0, 0), 255, reason: '모서리는 외곽선');
    expect(_alpha(pixels, 2, 0), 255, reason: '윗변은 외곽선');
    expect(_alpha(pixels, 2, 2), (0.4 * 255).round(), reason: '가운데는 채움');
    // 채움 픽셀은 프리멀티플라이라 원색보다 어둡다
    expect(_pixel(pixels, 2, 2), isNot(_red));
  });

  test('외곽선 두께를 키우면 더 안쪽까지 선이 된다', () {
    final mask = List.generate(7, (_) => List.filled(7, 1.0));
    final thick = buildMaskPixels(
      [_det('can', 0.9, mask: mask)],
      colorOf: (_) => _red,
      threshold: 0.5,
      fillOpacity: 0.4,
      strokeWidth: 2,
    )!;
    expect(_alpha(thick, 1, 1), 255, reason: '두께 2 면 안쪽 한 칸도 외곽선');
    expect(_alpha(thick, 3, 3), (0.4 * 255).round());
  });

  test('크기가 다른 마스크는 좌표를 맞출 수 없어 건너뛴다', () {
    final pixels = buildMaskPixels(
      [
        _det('can', 0.9, mask: [
          [1.0, 1.0],
        ]),
        _det('paper', 0.8, mask: [
          [1.0],
          [1.0],
        ]),
      ],
      colorOf: (_) => _red,
      threshold: 0.5,
      fillOpacity: 0.4,
      strokeWidth: 1,
    )!;
    expect(Size(pixels.width.toDouble(), pixels.height.toDouble()),
        const Size(2, 1));
  });
}
