import 'package:flutter/material.dart';

import '../config/app_config.dart';
import '../data/recycling_guide.dart';
import '../services/detection_aggregator.dart';

/// 안내 패널에 표시할 문구 한 벌(아이콘·제목·설명).
class GuidePrompt {
  const GuidePrompt({
    required this.icon,
    required this.title,
    required this.subtitle,
    this.tips = const [],
  });

  final IconData icon;
  final String title;
  final String subtitle;

  /// 사용자가 바로 해 볼 수 있는 조치. 비어 있으면 표시하지 않습니다.
  final List<String> tips;

  static const loading = GuidePrompt(
    icon: Icons.hourglass_top_rounded,
    title: '모델을 불러오는 중입니다…',
    subtitle: '잠시만 기다려 주세요.',
  );

  static const liveEmpty = GuidePrompt(
    icon: Icons.center_focus_weak_rounded,
    title: '분리수거할 쓰레기를 화면에 담아주세요',
    subtitle: '쓰레기가 인식되면 버릴 곳을 알려드립니다.',
  );

  static const photoReady = GuidePrompt(
    icon: Icons.photo_camera_rounded,
    title: '쓰레기를 화면에 맞추고 촬영 버튼을 누르세요',
    subtitle: '사진 한 장을 찍어 버릴 곳을 알려드립니다.',
  );

  static const photoAnalyzing = GuidePrompt(
    icon: Icons.image_search_rounded,
    title: '사진을 분석하는 중입니다…',
    subtitle: '잠시만 기다려 주세요.',
  );

  static const photoNothing = GuidePrompt(
    icon: Icons.search_off_rounded,
    title: '사진에서 쓰레기를 찾지 못했어요',
    subtitle: '각도나 거리를 바꿔 다시 찍어 주세요.',
    tips: [
      '쓰레기가 화면 가운데에 크게 담기도록 한 걸음 다가가세요.',
      '위에서 비스듬히 내려다보는 각도로 바꿔 보세요.',
      '그림자·반사가 없도록 밝은 곳에서, 배경이 단순한 바닥에 두고 찍으세요.',
      '겹쳐 있으면 하나씩 떼어 놓고 찍으세요.',
      '그래도 안 되면 세부 옵션에서 신뢰도 임계값을 낮춰 보세요.',
    ],
  );

  static const photoFailed = GuidePrompt(
    icon: Icons.error_outline_rounded,
    title: '사진을 분석하지 못했어요',
    subtitle: '아래 "다시 찍기" 를 눌러 한 번 더 시도해 주세요.',
  );
}

/// 하단 안내 패널. 검출된 쓰레기를 어디에 버릴지 보여주거나,
/// 아무것도 없으면 [emptyPrompt] 를, 바쁠 때(모델 로딩·분석 중)는 [busyPrompt] 를 표시합니다.
class GuidePanel extends StatelessWidget {
  const GuidePanel({
    super.key,
    required this.guide,
    required this.detections,
    required this.maxItems,
    this.modelLoading = false,
    this.busyPrompt = GuidePrompt.loading,
    this.emptyPrompt = GuidePrompt.liveEmpty,
    this.regionId,
    this.regionNotice,
    this.onTapRegionNotice,
  });

  final RecyclingGuide guide;
  final List<ActiveDetection> detections;
  final int maxItems;

  /// true 면 검출 결과 대신 [busyPrompt] 를 보여줍니다.
  final bool modelLoading;
  final GuidePrompt busyPrompt;
  final GuidePrompt emptyPrompt;

  /// 적용할 지역 규칙 id. null 이면 공통 규칙만 씁니다.
  final String? regionId;

  /// 지역 전체 안내(배출 요일·수거 방식 등). 없으면 표시하지 않습니다.
  final String? regionNotice;

  /// 지역 안내 줄을 눌렀을 때. null 이면 누를 수 없습니다.
  final VoidCallback? onTapRegionNotice;

  @override
  Widget build(BuildContext context) {
    final bottomInset = MediaQuery.paddingOf(context).bottom;

    return Container(
      width: double.infinity,
      // 높이를 고정한다. 안내 내용에 따라 패널이 커지면 위 카메라 화면이 줄었다 늘었다 하기 때문.
      height: AppConfig.guidePanelHeight + bottomInset,
      padding: EdgeInsets.fromLTRB(16, 16, 16, 16 + bottomInset),
      decoration: const BoxDecoration(
        color: Color(0xE6101010),
        borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
      ),
      // 내용이 고정 높이를 넘으면 패널 안에서만 스크롤한다
      child: SingleChildScrollView(
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            if (regionNotice != null) ...[
              InkWell(
                onTap: onTapRegionNotice,
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Icon(
                      Icons.place_outlined,
                      size: 15,
                      color: Colors.lightBlueAccent,
                    ),
                    const SizedBox(width: 4),
                    Expanded(
                      child: Text(
                        regionNotice!,
                        style: const TextStyle(
                          color: Colors.lightBlueAccent,
                          fontSize: 12,
                        ),
                      ),
                    ),
                    if (onTapRegionNotice != null)
                      const Icon(
                        Icons.chevron_right_rounded,
                        size: 16,
                        color: Colors.lightBlueAccent,
                      ),
                  ],
                ),
              ),
              const SizedBox(height: 8),
            ],
            AnimatedSwitcher(
              duration: const Duration(milliseconds: 250),
              child: modelLoading
                  ? _EmptyPrompt(
                      key: ValueKey('busy-${busyPrompt.title}'),
                      prompt: busyPrompt,
                    )
                  : detections.isEmpty
                  ? _EmptyPrompt(
                      key: ValueKey('empty-${emptyPrompt.title}'),
                      prompt: emptyPrompt,
                    )
                  : _DetectionList(
                      key: ValueKey('list-${detections.length}'),
                      guide: guide,
                      detections: detections.take(maxItems).toList(),
                      regionId: regionId,
                    ),
            ),
          ],
        ),
      ),
    );
  }
}

class _EmptyPrompt extends StatelessWidget {
  const _EmptyPrompt({super.key, required this.prompt});

  final GuidePrompt prompt;

  @override
  Widget build(BuildContext context) {
    final text = Theme.of(context).textTheme;
    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Padding(
          padding: const EdgeInsets.only(top: 2),
          child: Icon(prompt.icon, size: 40, color: Colors.white70),
        ),
        const SizedBox(width: 14),
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            mainAxisSize: MainAxisSize.min,
            children: [
              Text(
                prompt.title,
                style: text.titleMedium?.copyWith(
                  color: Colors.white,
                  fontWeight: FontWeight.w700,
                ),
              ),
              const SizedBox(height: 4),
              Text(
                prompt.subtitle,
                style: text.bodyMedium?.copyWith(color: Colors.white60),
              ),
              // 조치 목록은 패널이 스크롤되므로 잘리지 않는다
              for (final tip in prompt.tips)
                Padding(
                  padding: const EdgeInsets.only(top: 6),
                  child: Row(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Padding(
                        padding: EdgeInsets.only(top: 5, right: 7),
                        child: Icon(
                          Icons.circle,
                          size: 5,
                          color: Colors.white38,
                        ),
                      ),
                      Expanded(
                        child: Text(
                          tip,
                          style: text.bodySmall?.copyWith(
                            color: Colors.white70,
                            height: 1.35,
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
            ],
          ),
        ),
      ],
    );
  }
}

class _DetectionList extends StatelessWidget {
  const _DetectionList({
    super.key,
    required this.guide,
    required this.detections,
    this.regionId,
  });

  final RecyclingGuide guide;
  final List<ActiveDetection> detections;
  final String? regionId;

  @override
  Widget build(BuildContext context) {
    return Column(
      mainAxisSize: MainAxisSize.min,
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        for (var i = 0; i < detections.length; i++) ...[
          if (i > 0) const SizedBox(height: 10),
          _buildCard(detections[i], primary: i == 0),
        ],
      ],
    );
  }

  Widget _buildCard(ActiveDetection detection, {required bool primary}) {
    final category = guide.resolveFor(detection.className, regionId: regionId);
    final altName = detection.alternativeClassName;
    final alternative = altName == null
        ? null
        : guide.resolveFor(altName, regionId: regionId);
    return _GuideCard(
      // 상태별 안내는 첫 번째(가장 확실한) 품목에만 보여 패널이 너무 길어지지 않게 한다
      conditionLines: primary ? _conditionLines(category) : const [],
      detection: detection,
      category: category,
      // 후보가 같은 분리수거 분류면 안내가 같으므로 표시하지 않는다
      alternativeCategory: alternative == null || alternative.id == category.id
          ? null
          : alternative,
      primary: primary,
    );
  }

  /// "코팅되어 있거나 영수증(감열지)이면 → 일반쓰레기" 형태의 안내 문구를 만든다.
  List<String> _conditionLines(RecyclingCategory category) {
    final lines = <String>[];
    for (final c in category.conditions.take(AppConfig.maxGuideConditions)) {
      final targetId = c.thenCategoryId;
      final target = targetId == null ? null : guide.categoryById(targetId);
      final buffer = StringBuffer(c.when);
      if (target != null) buffer.write(' → ${target.name}');
      final note = c.note;
      if (note != null && note.isNotEmpty) {
        buffer.write(target != null ? ' ($note)' : ' → $note');
      }
      lines.add(buffer.toString());
    }
    return lines;
  }
}

class _GuideCard extends StatelessWidget {
  const _GuideCard({
    required this.detection,
    required this.category,
    required this.primary,
    this.alternativeCategory,
    this.conditionLines = const [],
  });

  final ActiveDetection detection;
  final RecyclingCategory category;
  final bool primary;

  /// 신뢰도가 비슷하게 나온 다른 분류 후보. 없으면 null.
  final RecyclingCategory? alternativeCategory;

  /// 상태에 따라 다르게 버려야 하는 경우 안내(예: 코팅·오염된 종이는 일반쓰레기).
  final List<String> conditionLines;

  @override
  Widget build(BuildContext context) {
    final text = Theme.of(context).textTheme;
    final color = Color(category.color);
    final pct = (detection.confidence * 100).round();

    return Container(
      padding: EdgeInsets.all(primary ? 14 : 10),
      decoration: BoxDecoration(
        color: color.withValues(alpha: primary ? 0.22 : 0.12),
        border: Border.all(color: color, width: primary ? 2 : 1),
        borderRadius: BorderRadius.circular(14),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Container(
            width: primary ? 52 : 40,
            height: primary ? 52 : 40,
            decoration: BoxDecoration(color: color, shape: BoxShape.circle),
            child: Icon(
              categoryIcon(category.icon),
              color: Colors.white,
              size: primary ? 30 : 22,
            ),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Expanded(
                      child: Text(
                        '${category.name} → ${category.bin}',
                        style: (primary ? text.titleMedium : text.titleSmall)
                            ?.copyWith(
                              color: Colors.white,
                              fontWeight: FontWeight.w800,
                            ),
                      ),
                    ),
                    Text(
                      '${detection.className} $pct%',
                      style: text.labelSmall?.copyWith(color: Colors.white54),
                    ),
                  ],
                ),
                if (primary) ...[
                  const SizedBox(height: 4),
                  Text(
                    category.instruction,
                    style: text.bodyMedium?.copyWith(color: Colors.white),
                  ),
                ],
                if (alternativeCategory != null) ...[
                  const SizedBox(height: 6),
                  Row(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Icon(
                        Icons.help_outline_rounded,
                        size: 16,
                        color: Colors.amberAccent,
                      ),
                      const SizedBox(width: 4),
                      Expanded(
                        child: Text(
                          '또는 ${alternativeCategory!.name} → ${alternativeCategory!.bin} 일 수 있어요 '
                          '(${detection.alternativeClassName} '
                          '${((detection.alternativeConfidence ?? 0) * 100).round()}%)',
                          style: text.bodySmall?.copyWith(
                            color: Colors.amberAccent,
                          ),
                        ),
                      ),
                    ],
                  ),
                ],
                if (conditionLines.isNotEmpty) ...[
                  const SizedBox(height: 8),
                  // 상태에 따라 다르게 버려야 하는 경우 (예: 코팅·오염된 종이 → 일반쓰레기)
                  for (final line in conditionLines)
                    Padding(
                      padding: const EdgeInsets.only(top: 3),
                      child: Row(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          const Icon(
                            Icons.subdirectory_arrow_right_rounded,
                            size: 15,
                            color: Colors.white54,
                          ),
                          const SizedBox(width: 4),
                          Expanded(
                            child: Text(
                              line,
                              style: text.bodySmall?.copyWith(
                                color: Colors.white70,
                              ),
                            ),
                          ),
                        ],
                      ),
                    ),
                ],
              ],
            ),
          ),
        ],
      ),
    );
  }

  static IconData categoryIcon(String key) {
    switch (key) {
      case 'pet':
      case 'plastic':
        return Icons.local_drink_rounded;
      case 'vinyl':
        return Icons.shopping_bag_rounded;
      case 'can':
        return Icons.hardware_rounded;
      case 'glass':
        return Icons.wine_bar_rounded;
      case 'paper':
        return Icons.description_rounded;
      case 'cardboard':
        return Icons.inventory_2_rounded;
      case 'carton':
        return Icons.breakfast_dining_rounded;
      case 'styrofoam':
        return Icons.takeout_dining_rounded;
      case 'food':
        return Icons.restaurant_rounded;
      case 'ewaste':
        return Icons.battery_alert_rounded;
      case 'clothing':
        return Icons.checkroom_rounded;
      default:
        return Icons.delete_rounded;
    }
  }
}
