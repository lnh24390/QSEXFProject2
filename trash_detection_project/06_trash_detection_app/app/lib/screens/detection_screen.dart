import 'dart:async';
import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:gal/gal.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:ultralytics_yolo/ultralytics_yolo.dart';

import '../config/app_config.dart';
import '../config/detection_settings.dart';
import '../data/detection_mode.dart';
import '../data/recycling_guide.dart';
import '../services/confidence_policy.dart';
import '../services/detection_aggregator.dart';
import '../services/model_catalog.dart';
import '../services/mask_image.dart';
import '../services/photo_analysis.dart';
import '../services/photo_analyzer.dart';
import '../services/region_service.dart';
import '../widgets/detection_settings_sheet.dart';
import '../widgets/guide_panel.dart';
import '../widgets/live_mask_overlay.dart';
import '../widgets/model_picker_sheet.dart';
import '../widgets/region_info_sheet.dart';
import '../widgets/photo_result_view.dart';

/// 메인 화면: 위 카메라(YOLOView) / 촬영 사진 결과, 아래 분리수거 안내 패널.
///
/// 실시간 모드는 프레임마다 검출해 안내하고, 사진 모드는 촬영한 한 장을 분석해 안내합니다.
/// 두 모드 모두 같은 YOLOView 를 유지해 모드 전환 시 카메라·모델을 다시 띄우지 않습니다.
class DetectionScreen extends StatefulWidget {
  const DetectionScreen({super.key});

  @override
  State<DetectionScreen> createState() => _DetectionScreenState();
}

class _DetectionScreenState extends State<DetectionScreen>
    with WidgetsBindingObserver {
  final _controller = YOLOViewController();
  final _aggregator = DetectionAggregator();
  final _photoAnalyzer = PhotoAnalyzer();

  RecyclingGuide? _guide;
  ModelCatalog? _catalog;
  String? _modelPath;
  bool _modelLoading = true;
  String? _modelError;
  PermissionStatus? _cameraPermission;
  DetectionSettings _settings = DetectionSettings.defaults;
  DetectionMode _mode = AppConfig.defaultMode;

  // 실시간 모드
  List<ActiveDetection> _visible = const [];
  double _fps = 0;
  Timer? _refreshTimer;

  /// 로드된 모델이 세그멘테이션인지. 모델 메타데이터에서 직접 읽는다.
  ///
  /// YOLOView.onModelLoad 가 돌려주는 task 는 **우리가 넘긴 값**(지금은 null)이라 쓸 수 없다.
  bool _modelIsSegment = false;

  /// 실시간 마스크 이미지(마스크만 그리는 모드일 때만). 새 프레임이 오면 이전 것을 해제한다.
  ui.Image? _liveMask;

  /// 앞 프레임 마스크를 만드는 중인지. 밀리면 그 사이 프레임은 건너뛴다.
  bool _liveMaskBusy = false;

  // 사진 모드
  Uint8List? _capturedBytes;
  PhotoAnalysis? _photo;
  List<PhotoDetection> _photoDetections = const [];
  List<ActiveDetection> _photoItems = const [];
  /// 세그멘테이션 마스크 이미지(세그 모델일 때만). 새 결과가 오면 이전 것을 해제한다.
  ui.Image? _maskImage;
  bool _photoBusy = false;
  String? _photoError;
  // 모드 전환·다시 찍기·재분석으로 무효가 된 분석 결과를 버리기 위한 요청 번호
  int _photoRequestId = 0;

  /// 실시간 화면 캡처를 갤러리에 저장하는 중인지.
  bool _savingCapture = false;

  /// 현재 적용 중인 지역 규칙. null 이면 공통 규칙만 씁니다.
  RecyclingRegion? _region;

  /// GPS 로 읽은 행정구역 이름(설정 화면에 보여줌).
  String? _detectedRegionName;

  /// 지역 판별 실패 사유(권한·위치 꺼짐·시간 초과 등). 성공하면 null.
  String? _regionError;

  /// 실제로 적용할 지역. 판별 전·실패 시에도 공통 규칙은 보여준다.
  RecyclingRegion? get _effectiveRegion => _region ?? _guide?.fallbackRegion;

  bool get _isPhotoMode => _mode == DetectionMode.photo;

  /// 실시간 모드에서 박스 없이 마스크만 그려야 하는 상태인지.
  ///
  /// 플러그인 네이티브 오버레이는 박스와 마스크를 항상 함께 그리고 박스만 끄는 기능이 없어,
  /// 이 경우 오버레이를 통째로 끄고 마스크를 앱에서 직접 그린다.
  bool get _liveMasksOnly =>
      !_isPhotoMode &&
      _modelIsSegment &&
      !_settings.segmentShowBoxes;

  /// 현재 설정·모델 기준 신뢰도 정책. 클래스별 최적 임계값이 모델마다 달라 모델을 바꾸면 함께 바뀐다.
  ConfidencePolicy get _policy =>
      ConfidencePolicy.fromSettings(_settings, modelPath: _modelPath);

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _bootstrap();
  }

  Future<void> _bootstrap() async {
    final permission = await Permission.camera.request();
    final results = await Future.wait([
      RecyclingGuide.load(),
      ModelCatalog.load(),
      DetectionSettings.load(),
      DetectionSettings.loadModelAsset(),
    ]);
    if (!mounted) return;
    final guide = results[0] as RecyclingGuide;
    final catalog = results[1] as ModelCatalog;
    final settings = results[2] as DetectionSettings;
    final savedModel = results[3] as String?;
    setState(() {
      _cameraPermission = permission;
      _guide = guide;
      _catalog = catalog;
      _settings = settings;
      _modelPath ??= catalog.initialModel(preferred: savedModel);
    });
    _aggregator.holdTime = settings.detectionHoldTime;
    _startRefreshTimer();
    _applyRegion(settings, guide);
  }

  /// 설정 화면에 보여줄 지역 판별 상태 한 줄.
  String _regionStatusText() {
    if (_settings.regionId != null) return '직접 고른 지역을 씁니다.';
    if (_regionError != null) return '위치 확인 실패($_regionError) → 공통 규칙 사용';
    final name = _detectedRegionName;
    if (name == null) return '위치 확인 중… 확인 전에는 공통 규칙을 씁니다.';
    return '현재 위치: $name';
  }

  /// 지역별 분리수거 안내를 자세히 읽는 화면을 연다.
  void _openRegionInfo() {
    final guide = _guide;
    final region = _effectiveRegion;
    if (guide == null || region == null) return;
    unawaited(showRegionInfoSheet(
      context,
      guide: guide,
      region: region,
      detectedName: _detectedRegionName ?? _regionError,
    ));
  }

  /// 설정의 지역을 반영한다. 자동(null)이면 GPS 로 판별을 시도한다.
  void _applyRegion(DetectionSettings settings, RecyclingGuide guide) {
    final id = settings.regionId;
    if (id != null) {
      setState(() {
        _region = guide.regionById(id);
        _detectedRegionName = null;
        _regionError = null;
      });
      return;
    }
    unawaited(() async {
      final result = await RegionService.detect(guide);
      if (!mounted) return;
      // 자동 설정이 그대로일 때만 반영한다(그 사이 사용자가 직접 골랐을 수 있음)
      if (_settings.regionId != null) return;
      setState(() {
        _region = result.region;
        _detectedRegionName = result.administrativeName;
        _regionError = result.error;
      });
      debugPrint(
        'region lookup: region=${result.region?.id} '
        'name=${result.administrativeName} error=${result.error}',
      );
    }());
  }

  /// 권한 재요청으로 _bootstrap 이 다시 불리거나 갱신 주기가 바뀌어도 타이머가 하나만 돌게 한다.
  void _startRefreshTimer() {
    _refreshTimer?.cancel();
    _refreshTimer =
        Timer.periodic(_settings.uiRefreshInterval, (_) => _refreshVisible());
  }

  void _refreshVisible() {
    if (!mounted || _isPhotoMode) return;
    final next = _aggregator.snapshot();
    // 비어 있음 → 비어 있음 전환은 다시 그릴 필요가 없음
    if (next.isEmpty && _visible.isEmpty) return;
    setState(() => _visible = next);
  }

  /// 안내 대상 여부: 쓰레기가 아닌 클래스(사람·차량 등)는 빼고, 클래스별 신뢰도 기준을 적용한다.
  bool _accepts(RecyclingGuide guide, ConfidencePolicy policy, YOLOResult r) =>
      !guide.isIgnored(r.className) && policy.accepts(r);

  void _onResult(List<YOLOResult> results) {
    final guide = _guide;
    // 사진 모드에서는 미리보기 프레임 결과를 안내에 쓰지 않는다
    if (guide == null || _isPhotoMode) return;
    final policy = _policy;
    _aggregator.addFrame(
      results.where((r) => _accepts(guide, policy, r)).toList(),
    );
    if (_liveMasksOnly) unawaited(_updateLiveMask(results, guide));
  }

  void _onPerformance(YOLOPerformanceMetrics metrics) {
    _fps = metrics.fps;
  }

  void _applyThresholds() {
    _controller.setThresholds(
      // 네이티브는 임계값 하나만 받으므로 클래스별 기준 중 가장 낮은 값으로 받고 앱에서 다시 거른다
      confidenceThreshold: _policy.nativeThreshold,
      iouThreshold: _settings.iouThreshold,
      numItemsThreshold: _settings.maxDetections,
    );
  }

  /// 모델 파일의 메타데이터를 읽어 세그멘테이션 모델인지 확인한다.
  ///
  /// 확인되면 오버레이 모드를 다시 맞춘다(세그 + 박스끄기 → 앱이 마스크만 그림).
  Future<void> _updateModelIsSegment(String modelPath) async {
    var isSegment = false;
    try {
      final meta = await YOLO.inspectModel(modelPath);
      isSegment = (meta['task'] as String?)?.toLowerCase() == 'segment';
    } catch (e) {
      debugPrint('모델 task 확인 실패: $e');
    }
    if (!mounted || isSegment == _modelIsSegment) return;
    setState(() => _modelIsSegment = isSegment);
    _applyOverlayMode();
  }

  /// 네이티브 오버레이와 스트리밍 설정을 현재 모드·모델·설정에 맞춘다.
  ///
  /// 마스크만 그리는 모드에서는 마스크 데이터가 필요하므로 스트리밍에 마스크를 포함시킨다
  /// (기본 minimal 설정은 속도를 위해 마스크를 빼고 보낸다).
  void _applyOverlayMode() {
    final masksOnly = _liveMasksOnly;
    unawaited(_controller.setShowOverlays(!_isPhotoMode && !masksOnly));
    // 스트리밍 설정(마스크 포함 여부)은 뷰를 만들 때 정한다.
    // 실행 중에 setStreamingConfig 로 바꾸면 네이티브에서 결과 전송이 멈추는 현상이 있었다.
    if (!masksOnly) _clearLiveMask();
  }

  void _clearLiveMask() {
    if (_liveMask == null) return;
    final old = _liveMask;
    _liveMask = null;
    if (mounted) setState(() {});
    old?.dispose();
  }

  /// 실시간 프레임의 마스크 이미지를 만든다. 그리는 중이면 그 프레임은 건너뛴다.
  Future<void> _updateLiveMask(
      List<YOLOResult> results, RecyclingGuide guide) async {
    if (_liveMaskBusy) return;
    _liveMaskBusy = true;
    try {
      final policy = _policy;
      final accepted = results
          .where((r) => _accepts(guide, policy, r))
          .map((r) => PhotoDetection(result: r))
          .toList();
      final image = await buildMaskImage(
        accepted,
        colorOf: (d) => guide.resolve(d.className).color,
        threshold: AppConfig.maskThreshold,
        fillOpacity: AppConfig.maskOpacity,
        strokeWidth: AppConfig.maskStrokeWidth,
      );
      if (!mounted || !_liveMasksOnly) {
        image?.dispose();
        return;
      }
      final old = _liveMask;
      setState(() => _liveMask = image);
      old?.dispose();
    } finally {
      _liveMaskBusy = false;
    }
  }

  // ---------------------------------------------------------------- 모드

  void _setMode(DetectionMode mode) {
    if (mode == _mode) return;
    final wasShowingPhoto = _capturedBytes != null;
    setState(() {
      _mode = mode;
      _clearPhoto();
      _visible = const [];
    });
    _aggregator.clear();
    // 사진 모드 미리보기에는 실시간 박스를 그리지 않는다
    _applyOverlayMode();
    if (wasShowingPhoto) unawaited(_controller.resume());
  }

  /// setState 안에서 호출. 사진 모드 상태를 비우고 진행 중인 분석 결과를 무효화한다.
  void _clearPhoto() {
    _photoRequestId++;
    _capturedBytes = null;
    _photo = null;
    _photoDetections = const [];
    _photoItems = const [];
    _maskImage?.dispose();
    _maskImage = null;
    _photoBusy = false;
    _photoError = null;
  }

  // ---------------------------------------------------------------- 사진 모드

  Future<void> _capture() async {
    if (_photoBusy || _modelLoading || _modelPath == null) return;
    setState(() {
      _photoBusy = true;
      _photoError = null;
    });
    Uint8List? bytes;
    try {
      bytes = await _controller.capturePhoto(withOverlays: false);
    } catch (e) {
      debugPrint('capturePhoto failed: $e');
    }
    if (!mounted || !_isPhotoMode) return;
    if (bytes == null || bytes.isEmpty) {
      setState(() {
        _photoBusy = false;
        _photoError = '사진을 찍지 못했습니다. 다시 시도해 주세요.';
      });
      return;
    }
    setState(() => _capturedBytes = bytes);
    // 결과를 보는 동안 카메라·실시간 추론을 멈춰 배터리와 발열을 줄인다
    unawaited(_controller.pause());
    await _analyzeCaptured();
  }

  /// 촬영한 사진을 현재 모델·설정으로 (다시) 분석한다.
  Future<void> _analyzeCaptured() async {
    final bytes = _capturedBytes;
    final modelPath = _modelPath;
    final guide = _guide;
    if (bytes == null || modelPath == null || guide == null) return;
    final requestId = ++_photoRequestId;
    final policy = _policy;
    final settings = _settings;
    setState(() {
      _photoBusy = true;
      _photoError = null;
    });
    try {
      final analysis = await _photoAnalyzer.analyze(
        bytes,
        modelPath: modelPath,
        confidenceThreshold: policy.nativeThreshold,
        // 플러그인 NMS 는 클래스를 구분하지 않아 다른 분류 후보를 지우므로, 느슨하게 받고 앱에서 합친다
        iouThreshold: AppConfig.photoNativeIouThreshold,
      );
      if (!mounted || requestId != _photoRequestId) return;
      final detections = refinePhotoResults(
        analysis.results,
        accept: (r) => _accepts(guide, policy, r),
        maxItems: settings.maxDetections,
        iouThreshold: settings.iouThreshold,
        ambiguousGap: AppConfig.ambiguousConfidenceGap,
        // 분리수거 분류가 같으면 안내가 같으므로 후보로 보지 않는다
        categoryOf: (className) => guide.resolve(className).id,
      );
      // 세그멘테이션 모델이면 마스크 이미지를 만든다(detect 모델은 마스크가 없어 null).
      final mask = await buildMaskImage(
        detections,
        colorOf: (d) => guide.resolve(d.className).color,
        threshold: AppConfig.maskThreshold,
        fillOpacity: AppConfig.maskOpacity,
        strokeWidth: AppConfig.maskStrokeWidth,
      );
      if (!mounted || requestId != _photoRequestId) {
        mask?.dispose();
        return;
      }
      setState(() {
        _photo = analysis;
        _photoDetections = detections;
        _photoItems = summarizeDetections(detections);
        _maskImage?.dispose();
        _maskImage = mask;
        _photoBusy = false;
      });
    } catch (e) {
      if (!mounted || requestId != _photoRequestId) return;
      setState(() {
        _photo = null;
        _photoDetections = const [];
        _photoItems = const [];
        _maskImage?.dispose();
        _maskImage = null;
        _photoBusy = false;
        _photoError = '사진 분석 실패: $e';
      });
    }
  }

  void _retake() {
    setState(_clearPhoto);
    unawaited(_controller.resume());
  }

  // ---------------------------------------------------------------- 실시간 화면 캡처

  /// 실시간 화면(검출 박스·마스크 포함)을 찍어 갤러리에 저장한다.
  Future<void> _captureScreen() async {
    if (_savingCapture || _modelLoading) return;
    setState(() => _savingCapture = true);
    try {
      final bytes = await _controller.capturePhoto(withOverlays: true);
      if (bytes == null || bytes.isEmpty) {
        throw StateError('카메라에서 화면을 가져오지 못했습니다');
      }
      if (!await Gal.hasAccess()) await Gal.requestAccess();
      await Gal.putImageBytes(bytes, album: AppConfig.captureAlbumName);
      _showSnack('갤러리(${AppConfig.captureAlbumName})에 저장했습니다');
    } catch (e) {
      _showSnack('저장 실패: $e');
    } finally {
      if (mounted) setState(() => _savingCapture = false);
    }
  }

  void _showSnack(String message) {
    if (!mounted) return;
    ScaffoldMessenger.of(context)
      ..clearSnackBars()
      ..showSnackBar(SnackBar(content: Text(message)));
  }

  // ---------------------------------------------------------------- 모델·옵션

  Future<void> _pickModel() async {
    final catalog = _catalog;
    final current = _modelPath;
    if (catalog == null || current == null) return;
    final chosen = await showModelPickerSheet(
      context,
      catalog: catalog,
      currentModel: current,
      onOpenSettings: _openSettings,
    );
    if (chosen == null || chosen == current || !mounted) return;
    // YOLOView 가 modelPath 변경을 감지해 직접 교체한다. controller.switchModel 을 따로 부르면
    // 교체가 두 번 일어나 연속 전환 시 순서가 꼬일 수 있으므로 부르지 않는다.
    // 모델별 임계값은 onModelLoad 의 _applyThresholds 에서 새 모델 기준으로 적용된다.
    setState(() {
      _modelLoading = true;
      _modelError = null;
      _modelPath = chosen;
    });
    _aggregator.clear();
    // 보고 있는 사진이 있으면 새 모델로 다시 분석
    if (_capturedBytes != null) unawaited(_analyzeCaptured());
  }

  Future<void> _openSettings() async {
    final modelPath = _modelPath;
    final next = await showDetectionSettingsSheet(
      context,
      current: _settings,
      modelName: modelPath == null ? null : ModelCatalog.displayName(modelPath),
      regions: _guide?.regions ?? const [],
      detectedRegionName: _detectedRegionName,
      onOpenRegionInfo: _effectiveRegion == null ? null : _openRegionInfo,
      regionStatus: _regionStatusText(),
    );
    if (next == null || next == _settings || !mounted) return;
    final refreshChanged = next.uiRefreshInterval != _settings.uiRefreshInterval;
    setState(() => _settings = next);
    _aggregator.holdTime = next.detectionHoldTime;
    if (refreshChanged) _startRefreshTimer();
    // 모델 로딩 중이면 onModelLoad 에서 적용된다
    if (!_modelLoading) {
      _applyThresholds();
      _applyOverlayMode();
    }
    unawaited(next.save());
    final guide = _guide;
    if (guide != null && next.regionId != _settings.regionId) {
      _applyRegion(next, guide);
    }
    // 보고 있는 사진이 있으면 바뀐 임계값·최대 개수로 다시 분석
    if (_capturedBytes != null) unawaited(_analyzeCaptured());
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (state == AppLifecycleState.paused) {
      // 백그라운드로 가면 남아 있던 검출 결과를 비워 복귀 시 오래된 안내가 보이지 않게 함
      _aggregator.clear();
    } else if (state == AppLifecycleState.resumed && _capturedBytes != null) {
      // 복귀 시 네이티브가 카메라를 다시 켤 수 있으므로, 사진 결과를 보는 중이면 다시 멈춘다
      unawaited(_controller.pause());
    }
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _refreshTimer?.cancel();
    _controller.dispose();
    _maskImage?.dispose();
    _liveMask?.dispose();
    unawaited(_photoAnalyzer.dispose());
    super.dispose();
  }

  // ---------------------------------------------------------------- UI

  String _statusText(String modelPath) {
    final photo = _photo;
    if (!_isPhotoMode) {
      return '${ModelCatalog.displayName(modelPath)} · ${_fps.toStringAsFixed(0)} fps';
    }
    if (photo == null) return '${ModelCatalog.displayName(modelPath)} · 사진 모드';
    return '${ModelCatalog.displayName(photo.modelPath)} · '
        '${photo.elapsed.inMilliseconds} ms · ${_photoDetections.length}개';
  }

  /// 사진 박스 라벨. 후보가 있으면 "종이/비닐? 70%" 처럼 표시한다.
  String _photoLabel(RecyclingGuide guide, PhotoDetection d) {
    final name = guide.resolveFor(d.className, regionId: _region?.id).name;
    final pct = (d.confidence * 100).round();
    final alternative = d.alternative;
    if (alternative == null) return '$name $pct%';
    final altName =
        guide.resolveFor(alternative.className, regionId: _region?.id).name;
    return '$name/$altName? $pct%';
  }

  Widget _buildGuidePanel(RecyclingGuide guide) {
    if (!_isPhotoMode) {
      return GuidePanel(
        guide: guide,
        detections: _visible,
        maxItems: _settings.maxGuideItems,
        modelLoading: _modelLoading,
        regionId: _effectiveRegion?.id,
        regionNotice: _effectiveRegion?.notice,
        onTapRegionNotice: _effectiveRegion == null ? null : _openRegionInfo,
      );
    }
    final hasPhoto = _capturedBytes != null;
    return GuidePanel(
      guide: guide,
      detections: hasPhoto ? _photoItems : const [],
      maxItems: _settings.maxGuideItems,
      modelLoading: _photoBusy || (!hasPhoto && _modelLoading),
      busyPrompt: _photoBusy ? GuidePrompt.photoAnalyzing : GuidePrompt.loading,
      emptyPrompt: _photoError != null
          ? GuidePrompt.photoFailed
          : hasPhoto
              ? GuidePrompt.photoNothing
              : GuidePrompt.photoReady,
      regionId: _effectiveRegion?.id,
      regionNotice: _effectiveRegion?.notice,
      onTapRegionNotice: _effectiveRegion == null ? null : _openRegionInfo,
    );
  }

  /// 실시간 모드 하단의 화면 캡처 버튼.
  Widget _buildLiveCaptureButton() {
    return Positioned(
      left: 0,
      right: 0,
      bottom: 20,
      child: Center(
        child: Tooltip(
          message: '화면 캡처',
          child: FilledButton.tonalIcon(
            onPressed: _savingCapture || _modelLoading ? null : _captureScreen,
            icon: _savingCapture
                ? const SizedBox(
                    width: 18,
                    height: 18,
                    child: CircularProgressIndicator(strokeWidth: 2.5),
                  )
                : const Icon(Icons.camera_alt_rounded),
            label: Text(_savingCapture ? '저장 중…' : '화면 캡처'),
          ),
        ),
      ),
    );
  }

  Widget _buildPhotoControls() {
    final bytes = _capturedBytes;
    return Positioned(
      left: 0,
      right: 0,
      bottom: 20,
      child: Center(
        child: bytes == null
            ? _ShutterButton(
                onPressed: _photoBusy || _modelLoading ? null : _capture,
              )
            : FilledButton.tonalIcon(
                onPressed: _photoBusy ? null : _retake,
                icon: const Icon(Icons.refresh_rounded),
                label: const Text('다시 찍기'),
              ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final guide = _guide;
    final catalog = _catalog;
    final modelPath = _modelPath;
    final permission = _cameraPermission;
    final ready = guide != null &&
        catalog != null &&
        modelPath != null &&
        permission != null &&
        permission.isGranted;

    Widget body;
    if (guide == null || catalog == null || permission == null) {
      body = const Center(child: CircularProgressIndicator());
    } else if (modelPath == null) {
      // 번들된 .tflite 가 하나도 없는 경우. 검출을 시작할 수 없다.
      body = const _NoModel();
    } else if (!permission.isGranted) {
      body = _PermissionDenied(
        permanentlyDenied: permission.isPermanentlyDenied,
        onRetry: _bootstrap,
      );
    } else {
      final bytes = _capturedBytes;
      final error = _modelError ?? (_isPhotoMode ? _photoError : null);
      body = Column(
        children: [
          Expanded(
            child: Stack(
              fit: StackFit.expand,
              children: [
                YOLOView(
                  // 마스크 스트리밍 여부가 바뀌면 키를 바꿔 뷰를 다시 만든다.
                  // (플러그인은 streamingConfig 프로퍼티 변경을 반영하지 않는다)
                  key: ValueKey('yolo-view-$_liveMasksOnly'),
                  modelPath: modelPath,
                  // task 미지정: 모델 메타데이터의 task(detect/segment)를 그대로 쓴다
                  controller: _controller,
                  cameraResolution: AppConfig.cameraResolution,
                  confidenceThreshold: _policy.nativeThreshold,
                  iouThreshold: _settings.iouThreshold,
                  // 마스크만 그리는 모드에서는 마스크 데이터가 필요하다
                  streamingConfig: _liveMasksOnly
                      ? const YOLOStreamingConfig.custom(includeMasks: true)
                      : const YOLOStreamingConfig.minimal(),
                  onResult: _onResult,
                  onPerformanceMetrics: _onPerformance,
                  onModelLoad: (path, task) {
                    if (!mounted) return;
                    // YOLOViewController.init() 이 플랫폼 뷰 부착 시점에 자기 기본값
                    // (conf 0.25 / iou 0.7 / numItems 30)을 setThresholds 로 밀어넣어
                    // YOLOView 생성 파라미터를 덮어쓴다. 현재 설정·모델 기준 값을 다시 적용한다.
                    // onModelLoad 의 task 는 우리가 넘긴 값(null)이라 쓰지 않고,
                    // 모델 메타데이터에서 직접 읽는다.
                    unawaited(_updateModelIsSegment(path));
                    _applyThresholds();
                    _applyOverlayMode();
                    // 사진 결과를 보는 중에 모델이 바뀌면 네이티브가 카메라를 다시 켤 수 있어 다시 멈춘다
                    if (_capturedBytes != null) unawaited(_controller.pause());
                    unawaited(DetectionSettings.saveModelAsset(path));
                    setState(() {
                      _modelLoading = false;
                      _modelError = null;
                    });
                  },
                  onModelError: (error, path, task) {
                    if (!mounted) return;
                    setState(() {
                      _modelLoading = false;
                      _modelError = '모델 로드 실패: $path\n$error';
                    });
                  },
                ),
                if (!_isPhotoMode && _liveMask != null)
                  Positioned.fill(child: LiveMaskOverlay(mask: _liveMask!)),
                if (_isPhotoMode && bytes != null)
                  PhotoResultView(
                    imageBytes: bytes,
                    imageSize: _photo?.imageSize,
                    detections: _photoDetections,
                    maskImage: _maskImage,
                    // 세그 모델(마스크 있음)은 기본적으로 마스크만 그린다.
                    // detect 모델은 마스크가 없으므로 항상 박스를 그린다.
                    showBoxes:
                        _maskImage == null || _settings.segmentShowBoxes,
                    colorFor: (d) => Color(guide.resolve(d.className).color),
                    labelFor: (d) => _photoLabel(guide, d),
                  ),
                if (_isPhotoMode && _photoBusy)
                  const Center(child: _BusyBadge(text: '분석 중…')),
                if (error != null)
                  Positioned(
                    left: 16,
                    right: 16,
                    top: 16,
                    child: Material(
                      color: Colors.red.shade900.withValues(alpha: 0.9),
                      borderRadius: BorderRadius.circular(10),
                      child: Padding(
                        padding: const EdgeInsets.all(12),
                        child: Text(error,
                            style: const TextStyle(color: Colors.white)),
                      ),
                    ),
                  ),
                Positioned(
                  left: 12,
                  // 사진 모드는 하단 중앙에 촬영 버튼이 있어 상태 표시를 위로 올린다
                  top: _isPhotoMode ? 12 : null,
                  bottom: _isPhotoMode ? null : 12,
                  child: _StatusChip(text: _statusText(modelPath)),
                ),
                if (_isPhotoMode)
                  _buildPhotoControls()
                else
                  _buildLiveCaptureButton(),
              ],
            ),
          ),
          _buildGuidePanel(guide),
        ],
      );
    }

    return Scaffold(
      appBar: AppBar(
        title: const Text('분리수거 도우미'),
        backgroundColor: Colors.black,
        actions: [
          IconButton(
            tooltip: '카메라 전환',
            icon: const Icon(Icons.cameraswitch_rounded),
            onPressed: _modelLoading || _capturedBytes != null || _photoBusy
                ? null
                : _controller.switchCamera,
          ),
          IconButton(
            tooltip: '모델·옵션 설정',
            icon: const Icon(Icons.tune_rounded),
            onPressed: _catalog == null ? null : _pickModel,
          ),
        ],
        bottom: PreferredSize(
          preferredSize: const Size.fromHeight(56),
          child: Padding(
            padding: const EdgeInsets.fromLTRB(16, 0, 16, 10),
            child: SizedBox(
              width: double.infinity,
              child: SegmentedButton<DetectionMode>(
                showSelectedIcon: false,
                segments: const [
                  ButtonSegment(
                    value: DetectionMode.live,
                    icon: Icon(Icons.videocam_rounded),
                    label: Text('실시간'),
                  ),
                  ButtonSegment(
                    value: DetectionMode.photo,
                    icon: Icon(Icons.photo_camera_rounded),
                    label: Text('사진'),
                  ),
                ],
                selected: {_mode},
                onSelectionChanged:
                    ready ? (selection) => _setMode(selection.first) : null,
              ),
            ),
          ),
        ),
      ),
      body: body,
    );
  }
}

class _StatusChip extends StatelessWidget {
  const _StatusChip({required this.text});
  final String text;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
      decoration: BoxDecoration(
        color: Colors.black54,
        borderRadius: BorderRadius.circular(20),
      ),
      child: Text(text,
          style: const TextStyle(color: Colors.white70, fontSize: 12)),
    );
  }
}

class _BusyBadge extends StatelessWidget {
  const _BusyBadge({required this.text});
  final String text;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 14),
      decoration: BoxDecoration(
        color: Colors.black87,
        borderRadius: BorderRadius.circular(14),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          const SizedBox(
            width: 20,
            height: 20,
            child: CircularProgressIndicator(strokeWidth: 2.5),
          ),
          const SizedBox(width: 12),
          Text(text, style: const TextStyle(color: Colors.white)),
        ],
      ),
    );
  }
}

/// 사진 모드의 원형 촬영 버튼.
class _ShutterButton extends StatelessWidget {
  const _ShutterButton({required this.onPressed});
  final VoidCallback? onPressed;

  @override
  Widget build(BuildContext context) {
    final enabled = onPressed != null;
    return Tooltip(
      message: '촬영',
      child: SizedBox(
        width: 76,
        height: 76,
        child: Material(
          color: enabled
              ? Colors.white.withValues(alpha: 0.9)
              : Colors.white.withValues(alpha: 0.3),
          shape: const CircleBorder(
            side: BorderSide(color: Colors.white, width: 4),
          ),
          child: InkWell(
            customBorder: const CircleBorder(),
            onTap: onPressed,
            child: Icon(
              Icons.photo_camera_rounded,
              size: 34,
              color: enabled ? Colors.black87 : Colors.black38,
            ),
          ),
        ),
      ),
    );
  }
}

/// `assets/models/` 에 .tflite 가 하나도 없을 때 표시하는 안내.
class _NoModel extends StatelessWidget {
  const _NoModel();

  @override
  Widget build(BuildContext context) {
    return const Center(
      child: Padding(
        padding: EdgeInsets.all(24),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(Icons.hide_source_rounded, size: 64, color: Colors.white54),
            SizedBox(height: 16),
            Text(
              '사용할 모델이 없습니다',
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.w700),
            ),
            SizedBox(height: 8),
            Text(
              'ml/scripts/export.py 로 변환한 .tflite 를\n'
              'app/assets/models/ 에 넣고 다시 빌드해 주세요.',
              textAlign: TextAlign.center,
              style: TextStyle(color: Colors.white70),
            ),
          ],
        ),
      ),
    );
  }
}

class _PermissionDenied extends StatelessWidget {
  const _PermissionDenied({
    required this.permanentlyDenied,
    required this.onRetry,
  });

  final bool permanentlyDenied;
  final VoidCallback onRetry;

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(24),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const Icon(Icons.no_photography_rounded,
                size: 64, color: Colors.white54),
            const SizedBox(height: 16),
            const Text(
              '카메라 권한이 필요합니다',
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.w700),
            ),
            const SizedBox(height: 8),
            const Text(
              '실시간으로 쓰레기를 인식하려면 카메라 접근을 허용해 주세요.',
              textAlign: TextAlign.center,
              style: TextStyle(color: Colors.white70),
            ),
            const SizedBox(height: 20),
            FilledButton(
              onPressed: permanentlyDenied ? openAppSettings : onRetry,
              child: Text(permanentlyDenied ? '설정에서 허용하기' : '다시 요청'),
            ),
          ],
        ),
      ),
    );
  }
}
