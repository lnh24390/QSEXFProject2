import 'dart:math' as math;

import 'package:ultralytics_yolo/ultralytics_yolo.dart';

import '../config/app_config.dart';
import '../config/detection_settings.dart';
import '../data/recycling_guide.dart';
import 'model_catalog.dart';

/// 모델별·클래스별 신뢰도 임계값 정책.
///
/// 플러그인(네이티브)은 임계값을 하나만 받으므로 [nativeThreshold](가장 낮은 기준)로 넉넉히 받은 뒤,
/// 앱에서 [accepts] 로 클래스마다 기준을 다시 적용합니다.
class ConfidencePolicy {
  const ConfidencePolicy({
    required this.globalThreshold,
    this.classThresholds = const {},
  });

  /// [modelPath] 의 파일명(확장자 제외)으로 [AppConfig.classConfidenceThresholds] 를 찾습니다.
  /// 설정에서 끄거나 측정값이 없는 모델이면 전체 임계값만 씁니다.
  factory ConfidencePolicy.fromSettings(
    DetectionSettings settings, {
    String? modelPath,
  }) {
    final perClass = settings.useClassThresholds && modelPath != null
        ? AppConfig.classConfidenceThresholds[ModelCatalog.displayName(modelPath)]
        : null;
    return ConfidencePolicy(
      globalThreshold: settings.confidenceThreshold,
      classThresholds: perClass ?? const {},
    );
  }

  /// [classThresholds] 에 없는 클래스에 쓰는 기준.
  final double globalThreshold;

  /// 정규화된 클래스명 → 임계값.
  final Map<String, double> classThresholds;

  double thresholdFor(String className) =>
      classThresholds[RecyclingGuide.normalizeClassName(className)] ??
      globalThreshold;

  /// 플러그인에 넘길 임계값. 클래스별 기준 중 가장 낮은 값까지 받아야 앱에서 다시 거를 수 있다.
  double get nativeThreshold => classThresholds.isEmpty
      ? globalThreshold
      : math.min(globalThreshold, classThresholds.values.reduce(math.min));

  bool accepts(YOLOResult result) =>
      result.confidence >= thresholdFor(result.className);
}
