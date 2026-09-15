import 'dart:ui';

import 'package:flutter_test/flutter_test.dart';
import 'package:trash_sorter/config/app_config.dart';
import 'package:trash_sorter/config/detection_settings.dart';
import 'package:trash_sorter/services/confidence_policy.dart';
import 'package:ultralytics_yolo/ultralytics_yolo.dart';

YOLOResult _r(String name, double conf) => YOLOResult(
      classIndex: 0,
      className: name,
      confidence: conf,
      boundingBox: Rect.zero,
      normalizedBox: Rect.zero,
    );

void main() {
  const policy = ConfidencePolicy(
    globalThreshold: 0.45,
    classThresholds: {'glass': 0.24, 'paper': 0.61},
  );

  test('클래스별 값이 있으면 그 값을, 없으면 전체 값을 쓴다 (클래스명 정규화)', () {
    expect(policy.thresholdFor('glass'), 0.24);
    expect(policy.thresholdFor(' Glass '), 0.24);
    expect(policy.thresholdFor('paper'), 0.61);
    expect(policy.thresholdFor('can'), 0.45);
  });

  test('플러그인에는 가장 낮은 기준을 넘긴다', () {
    expect(policy.nativeThreshold, 0.24);
    expect(const ConfidencePolicy(globalThreshold: 0.3).nativeThreshold, 0.3);
    expect(
      const ConfidencePolicy(globalThreshold: 0.1, classThresholds: {'glass': 0.24})
          .nativeThreshold,
      0.1,
    );
  });

  test('accepts: 클래스별 기준으로 거른다', () {
    expect(policy.accepts(_r('glass', 0.30)), isTrue);
    expect(policy.accepts(_r('paper', 0.50)), isFalse);
    expect(policy.accepts(_r('paper', 0.61)), isTrue);
    expect(policy.accepts(_r('can', 0.44)), isFalse);
  });

  group('fromSettings', () {
    const lnh = 'assets/models/lnh_yolov11n_50epoch_70000_.tflite';

    test('현재 모델의 측정값을 쓴다 (모델마다 다름)', () {
      final a = ConfidencePolicy.fromSettings(
        DetectionSettings.defaults,
        modelPath: lnh,
      );
      expect(
        a.thresholdFor('plastic'),
        AppConfig.classConfidenceThresholds['lnh_yolov11n_50epoch_70000_']!['plastic'],
      );
      final b = ConfidencePolicy.fromSettings(
        DetectionSettings.defaults,
        modelPath: 'assets/models/sj_yolov11n_epoch50_3000.tflite',
      );
      expect(
        b.thresholdFor('plastic'),
        AppConfig.classConfidenceThresholds['sj_yolov11n_epoch50_3000']!['plastic'],
      );
      expect(a.thresholdFor('plastic'), isNot(b.thresholdFor('plastic')));
    });

    test('측정값이 없는 모델이면 전체 임계값만 쓴다', () {
      final p = ConfidencePolicy.fromSettings(
        DetectionSettings.defaults.copyWith(confidenceThreshold: 0.35),
        modelPath: 'assets/models/unknown_model.tflite',
      );
      expect(p.classThresholds, isEmpty);
      expect(p.thresholdFor('glass'), 0.35);
      expect(p.nativeThreshold, 0.35);
    });

    test('설정에서 끄면 전체 임계값만 쓴다', () {
      final p = ConfidencePolicy.fromSettings(
        DetectionSettings.defaults.copyWith(
          useClassThresholds: false,
          confidenceThreshold: 0.4,
        ),
        modelPath: lnh,
      );
      expect(p.thresholdFor('glass'), 0.4);
      expect(p.nativeThreshold, 0.4);
    });
  });
}
