import 'package:flutter/material.dart';

import '../config/app_config.dart';
import '../config/detection_settings.dart';
import '../data/recycling_guide.dart';

/// "모델명 기준: glass 0.24 · vinyl 0.42 · …" 형태의 클래스별 임계값 요약(낮은 순).
String _classThresholdSummary(String? modelName) {
  final map =
      modelName == null ? null : AppConfig.classConfidenceThresholds[modelName];
  if (map == null || map.isEmpty) {
    return '현재 모델(${modelName ?? '-'})은 측정값이 없어 아래 전체 임계값을 씁니다.';
  }
  final entries = map.entries.toList()
    ..sort((a, b) => a.value.compareTo(b.value));
  final values =
      entries.map((e) => '${e.key} ${e.value.toStringAsFixed(2)}').join(' · ');
  return '$modelName 기준: $values';
}

/// "세부 옵션 설정" 바텀시트. [적용] 을 누르면 바뀐 설정을, 닫으면 null 을 반환합니다.
///
/// [modelName] 은 현재 모델 파일명(확장자 제외)으로, 클래스별 임계값 요약에 씁니다.
Future<DetectionSettings?> showDetectionSettingsSheet(
  BuildContext context, {
  required DetectionSettings current,
  String? modelName,
  List<RecyclingRegion> regions = const [],
  String? detectedRegionName,
  VoidCallback? onOpenRegionInfo,
  String? regionStatus,
}) {
  return showModalBottomSheet<DetectionSettings>(
    context: context,
    showDragHandle: true,
    isScrollControlled: true,
    builder: (context) => _DetectionSettingsSheet(
      initial: current,
      modelName: modelName,
      regions: regions,
      detectedRegionName: detectedRegionName,
      onOpenRegionInfo: onOpenRegionInfo,
      regionStatus: regionStatus,
    ),
  );
}

class _DetectionSettingsSheet extends StatefulWidget {
  const _DetectionSettingsSheet({
    required this.initial,
    this.modelName,
    this.regions = const [],
    this.detectedRegionName,
    this.onOpenRegionInfo,
    this.regionStatus,
  });

  final DetectionSettings initial;
  final String? modelName;

  /// 고를 수 있는 지역 목록(recycling_guide.json 의 regions).
  final List<RecyclingRegion> regions;

  /// GPS 로 판별된 지역 이름. 자동 선택지에 함께 보여줍니다.
  final String? detectedRegionName;

  /// 지역 안내를 자세히 읽는 화면을 여는 콜백. null 이면 버튼을 숨깁니다.
  final VoidCallback? onOpenRegionInfo;

  /// 지역 판별 상태 한 줄(현재 위치 / 확인 중 / 실패 사유).
  final String? regionStatus;

  @override
  State<_DetectionSettingsSheet> createState() =>
      _DetectionSettingsSheetState();
}

class _DetectionSettingsSheetState extends State<_DetectionSettingsSheet> {
  late DetectionSettings _s = widget.initial;

  @override
  Widget build(BuildContext context) {
    final maxHeight = MediaQuery.sizeOf(context).height * 0.85;
    return SafeArea(
      child: ConstrainedBox(
        constraints: BoxConstraints(maxHeight: maxHeight),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const Padding(
              padding: EdgeInsets.fromLTRB(20, 0, 20, 4),
              child: Align(
                alignment: Alignment.centerLeft,
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Text('세부 옵션 설정',
                        style: TextStyle(
                            fontSize: 18, fontWeight: FontWeight.w700)),
                    SizedBox(height: 2),
                    Text(
                      '[기본 최적화값] 은 측정으로 정한 권장 설정으로 되돌립니다. '
                      '바꾼 값은 [적용] 을 눌러야 반영됩니다.',
                      style: TextStyle(color: Colors.white60, fontSize: 12),
                    ),
                  ],
                ),
              ),
            ),
            Flexible(
              child: ListView(
                shrinkWrap: true,
                padding: const EdgeInsets.symmetric(vertical: 4),
                children: [
                  const _SectionHeader('검출'),
                  SwitchListTile(
                    contentPadding: const EdgeInsets.symmetric(horizontal: 20),
                    title: const Text('클래스별 신뢰도 임계값'),
                    subtitle: Text(
                      '모델마다 측정된 기준을 씁니다. ${_classThresholdSummary(widget.modelName)}',
                      style: const TextStyle(color: Colors.white60, fontSize: 12),
                    ),
                    value: _s.useClassThresholds,
                    onChanged: (v) =>
                        setState(() => _s = _s.copyWith(useClassThresholds: v)),
                  ),
                  _SliderTile(
                    title: '신뢰도 임계값',
                    description: _s.useClassThresholds
                        ? '클래스별 기준이 없는 모델·클래스에만 적용됩니다. 낮추면 더 많이, 높이면 확실한 것만 검출합니다.'
                        : '낮추면 더 많이, 높이면 확실한 것만 검출합니다.',
                    valueLabel: _s.confidenceThreshold.toStringAsFixed(2),
                    value: _s.confidenceThreshold,
                    min: 0.05,
                    max: 0.95,
                    divisions: 18,
                    onChanged: (v) => setState(() =>
                        _s = _s.copyWith(confidenceThreshold: _round2(v))),
                  ),
                  _SliderTile(
                    title: 'IoU 임계값 (NMS)',
                    description: '겹친 박스를 하나로 합치는 기준입니다. 낮추면 더 적극적으로 합칩니다.',
                    valueLabel: _s.iouThreshold.toStringAsFixed(2),
                    value: _s.iouThreshold,
                    min: 0.1,
                    max: 0.9,
                    divisions: 16,
                    onChanged: (v) => setState(
                        () => _s = _s.copyWith(iouThreshold: _round2(v))),
                  ),
                  _SliderTile(
                    title: '프레임당 최대 검출 개수',
                    description: '한 화면에서 찾을 물체의 최대 개수입니다.',
                    valueLabel: '${_s.maxDetections}개',
                    value: _s.maxDetections.toDouble(),
                    min: 1,
                    max: 30,
                    divisions: 29,
                    onChanged: (v) => setState(
                        () => _s = _s.copyWith(maxDetections: v.round())),
                  ),
                  SwitchListTile(
                    contentPadding: const EdgeInsets.symmetric(horizontal: 20),
                    title: const Text('세그 모델에서 박스도 그리기'),
                    subtitle: const Text(
                      '끄면 마스크(색칠)만 보여줍니다. 실시간·사진 모드 모두 적용됩니다. '
                      '실시간에서 끄면 앱이 마스크를 직접 그려 프레임이 조금 느려질 수 있습니다.',
                      style: TextStyle(color: Colors.white60, fontSize: 12),
                    ),
                    value: _s.segmentShowBoxes,
                    onChanged: (v) =>
                        setState(() => _s = _s.copyWith(segmentShowBoxes: v)),
                  ),
                  if (widget.regions.isNotEmpty) ...[
                    const _SectionHeader('지역'),
                    Padding(
                      padding: const EdgeInsets.fromLTRB(20, 6, 20, 0),
                      child: Text(
                        '지자체마다 배출 방법이 다릅니다. 자동은 현재 위치로 지역을 판별합니다.'
                        '${widget.regionStatus == null ? '' : '\n'}'
                        '${widget.regionStatus ?? ''}',
                        style: const TextStyle(
                          color: Colors.white60,
                          fontSize: 12,
                        ),
                      ),
                    ),
                    Padding(
                      padding: const EdgeInsets.fromLTRB(20, 8, 20, 0),
                      child: Wrap(
                        spacing: 8,
                        children: [
                          ChoiceChip(
                            label: const Text('자동(GPS)'),
                            selected: _s.regionId == null,
                            onSelected: (_) => setState(
                              () => _s = _s.copyWith(clearRegionId: true),
                            ),
                          ),
                          for (final r in widget.regions)
                            ChoiceChip(
                              label: Text(r.name),
                              selected: _s.regionId == r.id,
                              onSelected: (_) => setState(
                                () => _s = _s.copyWith(regionId: r.id),
                              ),
                            ),
                        ],
                      ),
                    ),
                  ],
                  if (widget.onOpenRegionInfo != null)
                    Padding(
                      padding: const EdgeInsets.fromLTRB(20, 4, 20, 0),
                      child: Align(
                        alignment: Alignment.centerLeft,
                        child: TextButton.icon(
                          onPressed: widget.onOpenRegionInfo,
                          icon: const Icon(Icons.menu_book_rounded, size: 18),
                          label: const Text('이 지역 분리수거 안내 읽기'),
                        ),
                      ),
                    ),
                  const _SectionHeader('안내 패널'),
                  _SliderTile(
                    title: '검출 유지 시간',
                    description: '물체가 안 보여도 이 시간 동안은 안내를 유지해 깜빡임을 막습니다.',
                    valueLabel: _seconds(_s.detectionHoldTime),
                    value: _s.detectionHoldTime.inMilliseconds.toDouble(),
                    min: 200,
                    max: 5000,
                    divisions: 48,
                    onChanged: (v) => setState(() => _s = _s.copyWith(
                        detectionHoldTime: Duration(milliseconds: v.round()))),
                  ),
                  _SliderTile(
                    title: '안내 갱신 주기',
                    description: '하단 안내를 다시 그리는 간격입니다. 짧을수록 빠르게 반응합니다.',
                    valueLabel: _seconds(_s.uiRefreshInterval),
                    value: _s.uiRefreshInterval.inMilliseconds.toDouble(),
                    min: 100,
                    max: 2000,
                    divisions: 38,
                    onChanged: (v) => setState(() => _s = _s.copyWith(
                        uiRefreshInterval: Duration(milliseconds: v.round()))),
                  ),
                  _SliderTile(
                    title: '동시에 표시할 품목 수',
                    description: '하단 패널에 한 번에 보여줄 안내 카드 개수입니다.',
                    valueLabel: '${_s.maxGuideItems}개',
                    value: _s.maxGuideItems.toDouble(),
                    min: 1,
                    max: 5,
                    divisions: 4,
                    onChanged: (v) => setState(
                        () => _s = _s.copyWith(maxGuideItems: v.round())),
                  ),
                ],
              ),
            ),
            Padding(
              padding: const EdgeInsets.fromLTRB(16, 8, 16, 12),
              child: Row(
                children: [
                  // AppConfig 의 권장값(측정으로 정한 클래스별 임계값 사용 포함)으로 되돌린다.
                  // 여기서는 화면의 값만 바뀌고, [적용] 을 눌러야 실제로 반영된다.
                  TextButton.icon(
                    onPressed: _s == DetectionSettings.defaults
                        ? null
                        : () =>
                            setState(() => _s = DetectionSettings.defaults),
                    icon: const Icon(Icons.restore_rounded, size: 18),
                    label: const Text('기본 최적화값'),
                  ),
                  const Spacer(),
                  TextButton(
                    onPressed: () => Navigator.of(context).pop(),
                    child: const Text('취소'),
                  ),
                  const SizedBox(width: 8),
                  FilledButton(
                    onPressed: () => Navigator.of(context).pop(_s),
                    child: const Text('적용'),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  static double _round2(double v) => (v * 100).roundToDouble() / 100;

  static String _seconds(Duration d) =>
      '${(d.inMilliseconds / 1000).toStringAsFixed(d.inMilliseconds % 100 == 0 ? 1 : 2)}초';
}

class _SectionHeader extends StatelessWidget {
  const _SectionHeader(this.text);
  final String text;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(20, 12, 20, 0),
      child: Text(
        text,
        style: TextStyle(
          color: Theme.of(context).colorScheme.primary,
          fontWeight: FontWeight.w700,
        ),
      ),
    );
  }
}

class _SliderTile extends StatelessWidget {
  const _SliderTile({
    required this.title,
    required this.description,
    required this.valueLabel,
    required this.value,
    required this.min,
    required this.max,
    required this.divisions,
    required this.onChanged,
  });

  final String title;
  final String description;
  final String valueLabel;
  final double value;
  final double min;
  final double max;
  final int divisions;
  final ValueChanged<double> onChanged;

  @override
  Widget build(BuildContext context) {
    final text = Theme.of(context).textTheme;
    return Padding(
      padding: const EdgeInsets.fromLTRB(20, 10, 20, 0),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Expanded(
                child: Text(title,
                    style: text.titleSmall
                        ?.copyWith(fontWeight: FontWeight.w600)),
              ),
              Text(valueLabel,
                  style: text.titleSmall?.copyWith(
                    color: Theme.of(context).colorScheme.primary,
                    fontWeight: FontWeight.w700,
                  )),
            ],
          ),
          Text(description,
              style: text.bodySmall?.copyWith(color: Colors.white60)),
          Slider(
            // 저장값이 범위를 벗어나도 슬라이더가 깨지지 않게 보정
            value: value.clamp(min, max),
            min: min,
            max: max,
            divisions: divisions,
            label: valueLabel,
            onChanged: onChanged,
          ),
        ],
      ),
    );
  }
}
