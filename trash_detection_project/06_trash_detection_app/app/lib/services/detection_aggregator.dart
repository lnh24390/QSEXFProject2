import 'package:ultralytics_yolo/ultralytics_yolo.dart';

import '../config/app_config.dart';

/// 화면에 현재 "보이는" 것으로 간주되는 품목 하나.
class ActiveDetection {
  ActiveDetection({
    required this.className,
    required this.confidence,
    required this.lastSeen,
    this.alternativeClassName,
    this.alternativeConfidence,
  });

  final String className;
  double confidence;
  DateTime lastSeen;
  int hits = 1;

  /// 같은 물체에서 신뢰도가 비슷하게 나온 다른 분류 후보(사진 모드). 없으면 null.
  String? alternativeClassName;
  double? alternativeConfidence;
}

/// 프레임마다 들어오는 검출 결과를 시간축으로 안정화합니다.
///
/// - 같은 클래스는 하나로 합칩니다.
/// - 한두 프레임 놓쳐도 [holdTime] 동안은 유지해 깜빡임을 막습니다.
/// - 누적 검출 횟수와 신뢰도로 정렬해 안내 우선순위를 정합니다.
class DetectionAggregator {
  DetectionAggregator({this.holdTime = AppConfig.detectionHoldTime});

  /// 마지막 검출 후 이 시간이 지나면 "사라진 것"으로 처리. 세부 옵션 설정에서 바뀔 수 있습니다.
  Duration holdTime;

  final Map<String, ActiveDetection> _active = {};

  void addFrame(List<YOLOResult> results, {DateTime? now}) {
    final t = now ?? DateTime.now();
    for (final r in results) {
      if (r.className.isEmpty) continue;
      final existing = _active[r.className];
      if (existing == null) {
        _active[r.className] = ActiveDetection(
          className: r.className,
          confidence: r.confidence,
          lastSeen: t,
        );
      } else {
        existing.lastSeen = t;
        existing.hits += 1;
        // 지수 이동 평균으로 신뢰도 부드럽게
        existing.confidence = existing.confidence * 0.7 + r.confidence * 0.3;
      }
    }
  }

  /// 유지 시간이 지난 항목을 제거하고 현재 활성 목록을 반환합니다.
  List<ActiveDetection> snapshot({DateTime? now}) {
    final t = now ?? DateTime.now();
    _active.removeWhere(
      (_, d) => t.difference(d.lastSeen) > holdTime,
    );
    final list = _active.values.toList()
      ..sort((a, b) {
        final byHits = b.hits.compareTo(a.hits);
        if (byHits != 0) return byHits;
        return b.confidence.compareTo(a.confidence);
      });
    return list;
  }

  bool get isEmpty => _active.isEmpty;

  void clear() => _active.clear();
}
