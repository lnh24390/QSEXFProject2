import 'package:flutter/foundation.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'app_config.dart';

/// 앱 안의 "세부 옵션 설정"에서 바꿀 수 있는 검출·안내 설정.
///
/// 기본값은 [AppConfig] 에서 가져오고, 사용자가 바꾼 값은 기기에 저장되어 다음 실행에도 유지됩니다.
@immutable
class DetectionSettings {
  const DetectionSettings({
    required this.confidenceThreshold,
    required this.useClassThresholds,
    required this.iouThreshold,
    required this.maxDetections,
    required this.detectionHoldTime,
    required this.uiRefreshInterval,
    required this.maxGuideItems,
    required this.segmentShowBoxes,
    required this.regionId,
  });

  static const defaults = DetectionSettings(
    confidenceThreshold: AppConfig.confidenceThreshold,
    useClassThresholds: AppConfig.useClassConfidenceThresholds,
    iouThreshold: AppConfig.iouThreshold,
    maxDetections: AppConfig.maxDetections,
    detectionHoldTime: AppConfig.detectionHoldTime,
    uiRefreshInterval: AppConfig.uiRefreshInterval,
    maxGuideItems: AppConfig.maxGuideItems,
    segmentShowBoxes: AppConfig.segmentShowBoxes,
    regionId: AppConfig.autoDetectRegion ? null : _defaultRegionId,
  );

  /// 자동 판별을 끌 때 쓰는 기본 지역 id (recycling_guide.json 의 regions 키).
  static const String _defaultRegionId = 'default';

  /// 전체 신뢰도 임계값. 클래스별 임계값을 쓰면 [AppConfig.classConfidenceThresholds] 에 없는 클래스에만 적용됩니다.
  final double confidenceThreshold;

  /// [AppConfig.classConfidenceThresholds] 를 적용할지 여부.
  final bool useClassThresholds;
  final double iouThreshold;
  final int maxDetections;
  final Duration detectionHoldTime;
  final Duration uiRefreshInterval;
  final int maxGuideItems;

  /// 사진 모드에서 세그멘테이션 마스크와 함께 박스도 그릴지 (실시간 모드에는 적용되지 않음).
  final bool segmentShowBoxes;

  /// 쓸 지역 규칙의 id. null 이면 GPS 로 자동 판별합니다.
  final String? regionId;

  DetectionSettings copyWith({
    double? confidenceThreshold,
    bool? useClassThresholds,
    double? iouThreshold,
    int? maxDetections,
    Duration? detectionHoldTime,
    Duration? uiRefreshInterval,
    int? maxGuideItems,
    bool? segmentShowBoxes,
    String? regionId,
    bool clearRegionId = false,
  }) {
    return DetectionSettings(
      confidenceThreshold: confidenceThreshold ?? this.confidenceThreshold,
      useClassThresholds: useClassThresholds ?? this.useClassThresholds,
      iouThreshold: iouThreshold ?? this.iouThreshold,
      maxDetections: maxDetections ?? this.maxDetections,
      detectionHoldTime: detectionHoldTime ?? this.detectionHoldTime,
      uiRefreshInterval: uiRefreshInterval ?? this.uiRefreshInterval,
      maxGuideItems: maxGuideItems ?? this.maxGuideItems,
      segmentShowBoxes: segmentShowBoxes ?? this.segmentShowBoxes,
      regionId: clearRegionId ? null : (regionId ?? this.regionId),
    );
  }

  static const _kConfidence = 'settings.confidenceThreshold';
  static const _kUseClassThresholds = 'settings.useClassThresholds';
  static const _kIou = 'settings.iouThreshold';
  static const _kMaxDetections = 'settings.maxDetections';
  static const _kHoldMs = 'settings.detectionHoldTimeMs';
  static const _kRefreshMs = 'settings.uiRefreshIntervalMs';
  static const _kMaxGuideItems = 'settings.maxGuideItems';
  static const _kSegmentShowBoxes = 'settings.segmentShowBoxes';
  static const _kRegionId = 'settings.regionId';
  static const _kModelAsset = 'settings.modelAsset';

  /// 저장된 설정을 읽습니다. 저장값이 없거나 읽기에 실패하면 [defaults] 를 씁니다.
  static Future<DetectionSettings> load() async {
    try {
      final prefs = SharedPreferencesAsync();
      const d = defaults;
      return DetectionSettings(
        confidenceThreshold:
            await prefs.getDouble(_kConfidence) ?? d.confidenceThreshold,
        useClassThresholds:
            await prefs.getBool(_kUseClassThresholds) ?? d.useClassThresholds,
        iouThreshold: await prefs.getDouble(_kIou) ?? d.iouThreshold,
        maxDetections: await prefs.getInt(_kMaxDetections) ?? d.maxDetections,
        detectionHoldTime: Duration(
          milliseconds:
              await prefs.getInt(_kHoldMs) ?? d.detectionHoldTime.inMilliseconds,
        ),
        uiRefreshInterval: Duration(
          milliseconds: await prefs.getInt(_kRefreshMs) ??
              d.uiRefreshInterval.inMilliseconds,
        ),
        maxGuideItems: await prefs.getInt(_kMaxGuideItems) ?? d.maxGuideItems,
        segmentShowBoxes:
            await prefs.getBool(_kSegmentShowBoxes) ?? d.segmentShowBoxes,
        // 빈 문자열 = 자동 판별(null). 저장값이 없으면 기본값을 쓴다.
        regionId: switch (await prefs.getString(_kRegionId)) {
          null => d.regionId,
          '' => null,
          final String id => id,
        },
      );
    } catch (e) {
      debugPrint('DetectionSettings.load failed, using defaults: $e');
      return defaults;
    }
  }

  Future<void> save() async {
    try {
      final prefs = SharedPreferencesAsync();
      await Future.wait([
        prefs.setDouble(_kConfidence, confidenceThreshold),
        prefs.setBool(_kUseClassThresholds, useClassThresholds),
        prefs.setDouble(_kIou, iouThreshold),
        prefs.setInt(_kMaxDetections, maxDetections),
        prefs.setInt(_kHoldMs, detectionHoldTime.inMilliseconds),
        prefs.setInt(_kRefreshMs, uiRefreshInterval.inMilliseconds),
        prefs.setInt(_kMaxGuideItems, maxGuideItems),
        prefs.setBool(_kSegmentShowBoxes, segmentShowBoxes),
        prefs.setString(_kRegionId, regionId ?? ''),
      ]);
    } catch (e) {
      debugPrint('DetectionSettings.save failed: $e');
    }
  }

  /// 마지막으로 로드에 성공한 모델 asset 경로. 없으면 null.
  static Future<String?> loadModelAsset() async {
    try {
      return await SharedPreferencesAsync().getString(_kModelAsset);
    } catch (e) {
      debugPrint('DetectionSettings.loadModelAsset failed: $e');
      return null;
    }
  }

  static Future<void> saveModelAsset(String modelAsset) async {
    try {
      await SharedPreferencesAsync().setString(_kModelAsset, modelAsset);
    } catch (e) {
      debugPrint('DetectionSettings.saveModelAsset failed: $e');
    }
  }

  @override
  bool operator ==(Object other) =>
      other is DetectionSettings &&
      other.confidenceThreshold == confidenceThreshold &&
      other.useClassThresholds == useClassThresholds &&
      other.iouThreshold == iouThreshold &&
      other.maxDetections == maxDetections &&
      other.detectionHoldTime == detectionHoldTime &&
      other.uiRefreshInterval == uiRefreshInterval &&
      other.maxGuideItems == maxGuideItems &&
      other.segmentShowBoxes == segmentShowBoxes &&
      other.regionId == regionId;

  @override
  int get hashCode => Object.hash(
        confidenceThreshold,
        useClassThresholds,
        iouThreshold,
        maxDetections,
        detectionHoldTime,
        uiRefreshInterval,
        maxGuideItems,
        segmentShowBoxes,
        regionId,
      );
}
