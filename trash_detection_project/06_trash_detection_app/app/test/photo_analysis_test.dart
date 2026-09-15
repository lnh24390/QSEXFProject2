import 'dart:ui';

import 'package:flutter_test/flutter_test.dart';
import 'package:trash_sorter/services/photo_analysis.dart';
import 'package:ultralytics_yolo/ultralytics_yolo.dart';

const _boxA = Rect.fromLTRB(0.1, 0.2, 0.3, 0.4);
const _boxB = Rect.fromLTRB(0.6, 0.6, 0.9, 0.9);

Map<String, dynamic> _det(String name, double conf, {Rect box = _boxA}) => {
      'classIndex': 0,
      'className': name,
      'confidence': conf,
      'boundingBox': {'left': 0.0, 'top': 0.0, 'right': 10.0, 'bottom': 10.0},
      'normalizedBox': {
        'left': box.left,
        'top': box.top,
        'right': box.right,
        'bottom': box.bottom,
      },
    };

YOLOResult _r(String name, double conf, {Rect box = _boxA}) =>
    YOLOResult.fromMap(_det(name, conf, box: box));

void main() {
  group('parseDetections', () {
    test('신뢰도 내림차순으로 정렬하고 maxItems 까지만 반환한다', () {
      final raw = <String, dynamic>{
        'detections': [_det('can', 0.5), _det('paper', 0.9), _det('glass', 0.7)],
      };
      final results = parseDetections(raw, maxItems: 2);
      expect(results.map((r) => r.className), ['paper', 'glass']);
      expect(results.first.normalizedBox.left, closeTo(0.1, 1e-9));
      expect(parseDetections(raw), hasLength(3));
    });

    test('형식이 잘못되거나 클래스명이 빈 항목은 버린다', () {
      final raw = <String, dynamic>{
        'detections': ['broken', _det('', 0.9), _det('can', 0.6)],
      };
      expect(parseDetections(raw).map((r) => r.className), ['can']);
    });

    test('detections 가 없으면 빈 목록', () {
      expect(parseDetections(<String, dynamic>{}), isEmpty);
    });
  });

  group('parseImageSize', () {
    test('imageSize 를 읽는다', () {
      final size = parseImageSize(<String, dynamic>{
        'imageSize': {'width': 3000, 'height': 4000},
      });
      expect(size?.width, 3000);
      expect(size?.height, 4000);
    });

    test('값이 없거나 0 이면 null', () {
      expect(parseImageSize(<String, dynamic>{}), isNull);
      expect(
        parseImageSize(<String, dynamic>{
          'imageSize': {'width': 0, 'height': 10},
        }),
        isNull,
      );
    });
  });

  test('boxIou', () {
    expect(boxIou(_boxA, _boxA), closeTo(1, 1e-9));
    expect(boxIou(_boxA, _boxB), 0);
    // 가로로 절반 겹침: 교집합 1, 합집합 3 → 1/3
    expect(
      boxIou(const Rect.fromLTRB(0, 0, 2, 1), const Rect.fromLTRB(1, 0, 3, 1)),
      closeTo(1 / 3, 1e-9),
    );
  });

  group('mergeOverlapping', () {
    test('같은 물체의 다른 분류 박스는 합치고, 신뢰도가 비슷하면 후보로 남긴다', () {
      final merged = mergeOverlapping(
        [_r('vinyl', 0.60), _r('paper', 0.70)],
        iouThreshold: 0.5,
        ambiguousGap: 0.15,
      );
      expect(merged, hasLength(1));
      expect(merged.single.className, 'paper');
      expect(merged.single.alternative?.className, 'vinyl');
    });

    test('신뢰도 차이가 크면 합치기만 하고 후보는 두지 않는다', () {
      final merged = mergeOverlapping(
        [_r('paper', 0.90), _r('vinyl', 0.50)],
        iouThreshold: 0.5,
        ambiguousGap: 0.15,
      );
      expect(merged, hasLength(1));
      expect(merged.single.alternative, isNull);
    });

    test('경계값(차이 = gap)도 후보로 인정한다', () {
      final merged = mergeOverlapping(
        [_r('paper', 0.80), _r('vinyl', 0.65)],
        iouThreshold: 0.5,
        ambiguousGap: 0.15,
      );
      expect(merged.single.alternative?.className, 'vinyl');
    });

    test('같은 분류(categoryOf)면 후보로 두지 않는다', () {
      final merged = mergeOverlapping(
        [_r('pet', 0.70), _r('plastic', 0.65)],
        iouThreshold: 0.5,
        ambiguousGap: 0.15,
        categoryOf: (c) => c == 'pet' ? 'plastic' : c,
      );
      expect(merged, hasLength(1));
      expect(merged.single.alternative, isNull);
    });

    test('겹치지 않는 박스는 따로 남는다', () {
      final merged = mergeOverlapping(
        [_r('paper', 0.70), _r('can', 0.60, box: _boxB)],
        iouThreshold: 0.5,
        ambiguousGap: 0.15,
      );
      expect(merged.map((d) => d.className), ['paper', 'can']);
    });
  });

  test('refinePhotoResults: 거르고 → 병합 → 개수 제한', () {
    const boxC = Rect.fromLTRB(0.0, 0.7, 0.1, 0.9);
    final input = [
      _r('paper', 0.70),
      _r('vinyl', 0.60),
      _r('can', 0.30, box: _boxB),
      _r('glass', 0.55, box: _boxB),
      _r('battery', 0.50, box: boxC),
    ];
    final refined = refinePhotoResults(
      input,
      accept: (r) => r.className != 'can',
      maxItems: 2,
      iouThreshold: 0.5,
      ambiguousGap: 0.15,
    );
    // can 은 걸러지고, paper·vinyl 이 먼저 합쳐진 뒤 2개로 자르므로 glass 가 살아남는다
    expect(refined.map((d) => d.className), ['paper', 'glass']);
    expect(refined.first.alternative?.className, 'vinyl');
  });

  test('summarizeDetections: 같은 클래스는 합치고 개수·신뢰도로 정렬, 후보를 전달한다', () {
    final items = summarizeDetections([
      PhotoDetection(result: _r('can', 0.6)),
      PhotoDetection(result: _r('paper', 0.95), alternative: _r('vinyl', 0.85)),
      PhotoDetection(result: _r('can', 0.8)),
    ]);
    expect(items.map((d) => d.className), ['can', 'paper']);
    expect(items.first.hits, 2);
    expect(items.first.confidence, closeTo(0.8, 1e-9));
    expect(items.first.alternativeClassName, isNull);
    expect(items.last.alternativeClassName, 'vinyl');
    expect(items.last.alternativeConfidence, closeTo(0.85, 1e-9));
  });
}
