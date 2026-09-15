import 'dart:typed_data';
import 'dart:ui';

import 'package:ultralytics_yolo/ultralytics_yolo.dart';

import 'detection_aggregator.dart';

/// 사진 한 장의 분석 결과.
class PhotoAnalysis {
  const PhotoAnalysis({
    required this.imageBytes,
    required this.imageSize,
    required this.results,
    required this.modelPath,
    required this.elapsed,
  });

  /// 촬영한 원본 JPEG.
  final Uint8List imageBytes;

  /// 모델에 들어간 이미지의 픽셀 크기. 박스 좌표(normalizedBox)의 기준.
  final Size imageSize;

  /// 플러그인이 돌려준 검출 결과 전체(신뢰도 내림차순). 클래스별 임계값·병합은 [refinePhotoResults] 에서.
  final List<YOLOResult> results;

  /// 분석에 사용한 모델 asset 경로.
  final String modelPath;

  /// 추론에 걸린 시간(모델 로드 제외).
  final Duration elapsed;
}

/// 사진 모드에서 화면·안내에 쓰는 검출 하나. 같은 물체에 겹친 박스를 합친 결과입니다.
class PhotoDetection {
  const PhotoDetection({required this.result, this.alternative});

  /// 합쳐진 박스 중 신뢰도가 가장 높은 박스.
  final YOLOResult result;

  /// 같은 물체에서 분류가 다르고 신뢰도가 비슷하게 나온 후보. 없으면 null.
  final YOLOResult? alternative;

  String get className => result.className;
  double get confidence => result.confidence;
  Rect get normalizedBox => result.normalizedBox;
}

/// `YOLO.predict` 가 돌려준 map 에서 검출 결과를 꺼냅니다.
///
/// 클래스명이 비었거나 형식이 잘못된 항목은 버리고 신뢰도 내림차순으로 정렬합니다.
/// [maxItems] 를 주면 그 개수까지만 반환합니다.
List<YOLOResult> parseDetections(Map<String, dynamic> raw, {int? maxItems}) {
  final list = raw['detections'];
  if (list is! List) return const [];
  final results = <YOLOResult>[];
  for (final item in list) {
    if (item is! Map) continue;
    try {
      final r = YOLOResult.fromMap(item);
      if (r.className.isNotEmpty) results.add(r);
    } catch (_) {
      // 형식이 맞지 않는 항목은 무시
    }
  }
  results.sort((a, b) => b.confidence.compareTo(a.confidence));
  return maxItems == null ? results : results.take(maxItems).toList();
}

/// `YOLO.predict` 결과의 `imageSize` 를 읽습니다. 없거나 잘못되면 null.
Size? parseImageSize(Map<String, dynamic> raw) {
  final size = raw['imageSize'];
  if (size is! Map) return null;
  final w = size['width'];
  final h = size['height'];
  if (w is! num || h is! num || w <= 0 || h <= 0) return null;
  return Size(w.toDouble(), h.toDouble());
}

/// 두 박스의 IoU(교집합 / 합집합).
double boxIou(Rect a, Rect b) {
  final inter = a.intersect(b);
  if (inter.width <= 0 || inter.height <= 0) return 0;
  final interArea = inter.width * inter.height;
  final union = a.width * a.height + b.width * b.height - interArea;
  return union <= 0 ? 0 : interArea / union;
}

/// 같은 물체에 겹친 박스를 클래스와 무관하게 하나로 합칩니다.
///
/// 신뢰도 높은 박스부터 보며 [iouThreshold] 이상 겹치는 박스를 흡수합니다. 흡수한 박스 중
/// 분류([categoryOf], 없으면 클래스명)가 다르고 신뢰도 차이가 [ambiguousGap] 이하인 첫 박스를 후보로 남깁니다.
List<PhotoDetection> mergeOverlapping(
  List<YOLOResult> results, {
  required double iouThreshold,
  required double ambiguousGap,
  String Function(String className)? categoryOf,
}) {
  final sorted = [...results]
    ..sort((a, b) => b.confidence.compareTo(a.confidence));
  final used = List<bool>.filled(sorted.length, false);
  String groupOf(String className) => categoryOf?.call(className) ?? className;

  final merged = <PhotoDetection>[];
  for (var i = 0; i < sorted.length; i++) {
    if (used[i]) continue;
    used[i] = true;
    final best = sorted[i];
    YOLOResult? alternative;
    for (var j = i + 1; j < sorted.length; j++) {
      if (used[j]) continue;
      final other = sorted[j];
      if (boxIou(best.normalizedBox, other.normalizedBox) < iouThreshold) {
        continue;
      }
      used[j] = true;
      // 부동소수 오차로 경계값(예: 0.80 - 0.65)이 빠지지 않게 약간의 여유를 둔다
      if (alternative == null &&
          groupOf(other.className) != groupOf(best.className) &&
          best.confidence - other.confidence <= ambiguousGap + 1e-9) {
        alternative = other;
      }
    }
    merged.add(PhotoDetection(result: best, alternative: alternative));
  }
  return merged;
}

/// 사진 분석 결과를 화면용으로 정리합니다.
///
/// [accept] 로 거르고 → 겹친 박스를 클래스 무관하게 합치고(= NMS, 후보 보존) → 신뢰도 순 [maxItems] 개.
/// 합치기 전에 개수를 자르면 같은 물체의 중복 박스가 자리를 차지하므로 반드시 병합 후에 자른다.
List<PhotoDetection> refinePhotoResults(
  List<YOLOResult> results, {
  required bool Function(YOLOResult result) accept,
  required int maxItems,
  required double iouThreshold,
  required double ambiguousGap,
  String Function(String className)? categoryOf,
}) {
  final merged = mergeOverlapping(
    results.where(accept).toList(),
    iouThreshold: iouThreshold,
    ambiguousGap: ambiguousGap,
    categoryOf: categoryOf,
  );
  return merged.take(maxItems).toList();
}

/// 안내 패널용 품목 목록을 만듭니다.
///
/// 같은 클래스는 하나로 합치고(hits = 개수, 신뢰도 = 가장 높은 값, 후보 = 신뢰도 높은 것부터 처음 나온 후보),
/// 개수가 많은 순 → 신뢰도가 높은 순으로 정렬합니다.
List<ActiveDetection> summarizeDetections(
  List<PhotoDetection> detections, {
  DateTime? now,
}) {
  final t = now ?? DateTime.now();
  final sorted = [...detections]
    ..sort((a, b) => b.confidence.compareTo(a.confidence));
  final byClass = <String, ActiveDetection>{};
  for (final d in sorted) {
    if (d.className.isEmpty) continue;
    final existing = byClass[d.className];
    if (existing == null) {
      byClass[d.className] = ActiveDetection(
        className: d.className,
        confidence: d.confidence,
        lastSeen: t,
        alternativeClassName: d.alternative?.className,
        alternativeConfidence: d.alternative?.confidence,
      );
    } else {
      existing.hits += 1;
      if (existing.alternativeClassName == null && d.alternative != null) {
        existing.alternativeClassName = d.alternative!.className;
        existing.alternativeConfidence = d.alternative!.confidence;
      }
    }
  }
  return byClass.values.toList()
    ..sort((a, b) {
      final byHits = b.hits.compareTo(a.hits);
      if (byHits != 0) return byHits;
      return b.confidence.compareTo(a.confidence);
    });
}
