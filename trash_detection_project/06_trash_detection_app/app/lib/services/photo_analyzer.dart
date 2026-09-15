import 'dart:async';
import 'dart:typed_data';
import 'dart:ui';

import 'package:ultralytics_yolo/ultralytics_yolo.dart';

import '../config/app_config.dart';
import 'photo_analysis.dart';

/// 촬영한 사진 한 장을 플러그인 `YOLO.predict` 로 분석합니다.
///
/// 실시간 `YOLOView` 와는 별도 모델 인스턴스라 서로 영향을 주지 않습니다.
/// 모델은 처음 분석할 때 로드하고, 다른 모델로 분석하면 이전 인스턴스를 해제하고 새로 로드합니다.
class PhotoAnalyzer {
  PhotoAnalyzer({this.maxCandidates = AppConfig.photoMaxCandidates});

  /// 플러그인에서 받을 최대 박스 수. 앱에서 겹친 박스를 합치기 전 개수라 넉넉히 둔다.
  final int maxCandidates;

  YOLO? _yolo;
  String? _loadedModel;

  // 연속 촬영·재분석이 모델 로드/교체와 겹치지 않도록 요청을 순서대로 처리한다.
  Future<void> _queue = Future<void>.value();

  /// [confidenceThreshold] 와 [iouThreshold] 는 플러그인에 넘기는 값입니다.
  /// 클래스별 기준과 최종 병합은 호출한 쪽에서 [refinePhotoResults] 로 적용합니다.
  Future<PhotoAnalysis> analyze(
    Uint8List imageBytes, {
    required String modelPath,
    required double confidenceThreshold,
    required double iouThreshold,
  }) {
    final task = _queue.then(
      (_) => _analyze(
        imageBytes,
        modelPath: modelPath,
        confidenceThreshold: confidenceThreshold,
        iouThreshold: iouThreshold,
      ),
    );
    _queue = task.then((_) {}, onError: (_) {});
    return task;
  }

  Future<PhotoAnalysis> _analyze(
    Uint8List imageBytes, {
    required String modelPath,
    required double confidenceThreshold,
    required double iouThreshold,
  }) async {
    final yolo = await _ensureModel(modelPath);
    final stopwatch = Stopwatch()..start();
    final raw = await yolo.predict(
      imageBytes,
      confidenceThreshold: confidenceThreshold,
      iouThreshold: iouThreshold,
    );
    stopwatch.stop();
    final size = parseImageSize(raw) ?? await _readImageSize(imageBytes);
    return PhotoAnalysis(
      imageBytes: imageBytes,
      imageSize: size,
      results: parseDetections(raw),
      modelPath: modelPath,
      elapsed: stopwatch.elapsed,
    );
  }

  Future<YOLO> _ensureModel(String modelPath) async {
    final current = _yolo;
    if (current != null && _loadedModel == modelPath) return current;
    await _disposeModel();
    final yolo = YOLO(
      modelPath: modelPath,
      // task 미지정: 모델 메타데이터로 detect/segment 자동 판별
      useMultiInstance: true,
      numItemsThreshold: maxCandidates,
    );
    final ok = await yolo.loadModel();
    if (!ok) {
      await yolo.dispose();
      throw StateError('사진 분석용 모델을 불러오지 못했습니다: $modelPath');
    }
    _yolo = yolo;
    _loadedModel = modelPath;
    return yolo;
  }

  Future<void> _disposeModel() async {
    final yolo = _yolo;
    _yolo = null;
    _loadedModel = null;
    if (yolo != null) await yolo.dispose();
  }

  /// 플러그인 결과에 이미지 크기가 없을 때, 전체 디코딩 없이 헤더만 읽어 크기를 구한다.
  static Future<Size> _readImageSize(Uint8List bytes) async {
    final buffer = await ImmutableBuffer.fromUint8List(bytes);
    try {
      final descriptor = await ImageDescriptor.encoded(buffer);
      final size =
          Size(descriptor.width.toDouble(), descriptor.height.toDouble());
      descriptor.dispose();
      return size;
    } finally {
      buffer.dispose();
    }
  }

  /// 진행 중인 요청이 끝난 뒤 모델 인스턴스를 해제합니다.
  Future<void> dispose() => _queue.then((_) => _disposeModel());
}
