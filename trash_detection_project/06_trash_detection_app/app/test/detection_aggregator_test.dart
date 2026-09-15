import 'dart:ui';

import 'package:flutter_test/flutter_test.dart';
import 'package:trash_sorter/config/app_config.dart';
import 'package:trash_sorter/services/detection_aggregator.dart';
import 'package:ultralytics_yolo/ultralytics_yolo.dart';

YOLOResult _r(String name, double conf) => YOLOResult(
      classIndex: 0,
      className: name,
      confidence: conf,
      boundingBox: Rect.zero,
      normalizedBox: Rect.zero,
    );

void main() {
  test('같은 클래스는 하나로 합쳐지고 hits 가 누적된다', () {
    final agg = DetectionAggregator();
    final t = DateTime(2026, 1, 1);
    agg.addFrame([_r('bottle', 0.9), _r('bottle', 0.8), _r('can', 0.7)], now: t);
    agg.addFrame([_r('bottle', 0.9)], now: t);
    final snap = agg.snapshot(now: t);
    expect(snap.map((d) => d.className), ['bottle', 'can']);
    expect(snap.first.hits, 3);
  });

  test('유지 시간이 지나면 사라진다', () {
    final agg = DetectionAggregator();
    final t = DateTime(2026, 1, 1);
    agg.addFrame([_r('bottle', 0.9)], now: t);
    expect(agg.snapshot(now: t.add(const Duration(milliseconds: 500))), hasLength(1));
    expect(
      agg.snapshot(now: t.add(AppConfig.detectionHoldTime * 2)),
      isEmpty,
    );
  });

  test('설정에서 바꾼 유지 시간을 따른다', () {
    final agg = DetectionAggregator(holdTime: const Duration(milliseconds: 300));
    final t = DateTime(2026, 1, 1);
    agg.addFrame([_r('bottle', 0.9)], now: t);
    expect(agg.snapshot(now: t.add(const Duration(milliseconds: 200))), hasLength(1));
    expect(agg.snapshot(now: t.add(const Duration(milliseconds: 400))), isEmpty);

    agg.holdTime = const Duration(seconds: 3);
    agg.addFrame([_r('can', 0.9)], now: t);
    expect(agg.snapshot(now: t.add(const Duration(seconds: 2))), hasLength(1));
  });

  test('빈 클래스명은 무시한다', () {
    final agg = DetectionAggregator();
    agg.addFrame([_r('', 0.9)]);
    expect(agg.isEmpty, isTrue);
  });
}
