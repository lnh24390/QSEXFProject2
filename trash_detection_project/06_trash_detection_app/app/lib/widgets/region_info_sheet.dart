import 'package:flutter/material.dart';
import 'package:url_launcher/url_launcher.dart';

import '../data/recycling_guide.dart';

/// 지역별 분리수거 안내를 자세히 읽는 바텀시트.
///
/// 내용은 `recycling_guide.json` 의 regions 에서 옵니다. 지자체 기준은 수시로 바뀌므로
/// 확인 시점(updated)과 출처(sources)를 함께 보여주고, 최종 확인은 거주지 공고로 안내합니다.
Future<void> showRegionInfoSheet(
  BuildContext context, {
  required RecyclingGuide guide,
  required RecyclingRegion region,
  String? detectedName,
}) {
  return showModalBottomSheet<void>(
    context: context,
    showDragHandle: true,
    isScrollControlled: true,
    builder: (context) => _RegionInfoSheet(
      guide: guide,
      region: region,
      detectedName: detectedName,
    ),
  );
}

class _RegionInfoSheet extends StatelessWidget {
  const _RegionInfoSheet({
    required this.guide,
    required this.region,
    this.detectedName,
  });

  final RecyclingGuide guide;
  final RecyclingRegion region;
  final String? detectedName;

  @override
  Widget build(BuildContext context) {
    final text = Theme.of(context).textTheme;
    final overrides = region.categoryOverrides;

    return SafeArea(
      child: ConstrainedBox(
        constraints: BoxConstraints(
          maxHeight: MediaQuery.sizeOf(context).height * 0.85,
        ),
        child: ListView(
          shrinkWrap: true,
          padding: const EdgeInsets.fromLTRB(20, 0, 20, 20),
          children: [
            Row(
              children: [
                const Icon(Icons.place_outlined, color: Colors.lightBlueAccent),
                const SizedBox(width: 8),
                Expanded(
                  child: Text(
                    region.name,
                    style: text.titleLarge?.copyWith(
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                ),
              ],
            ),
            if (detectedName != null)
              Padding(
                padding: const EdgeInsets.only(top: 4),
                child: Text(
                  '현재 위치: $detectedName',
                  style: text.bodySmall?.copyWith(color: Colors.white60),
                ),
              ),
            if (region.updated != null)
              Padding(
                padding: const EdgeInsets.only(top: 2),
                child: Text(
                  region.updated!,
                  style: text.bodySmall?.copyWith(color: Colors.white38),
                ),
              ),
            if (region.notice != null) ...[
              const SizedBox(height: 14),
              Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: Colors.lightBlue.withValues(alpha: 0.12),
                  borderRadius: BorderRadius.circular(10),
                ),
                child: Text(
                  region.notice!,
                  style: text.bodyMedium?.copyWith(
                    color: Colors.lightBlueAccent,
                  ),
                ),
              ),
            ],
            if (region.summary != null) ...[
              const SizedBox(height: 14),
              Text(region.summary!, style: text.bodyMedium),
            ],
            if (overrides.isNotEmpty) ...[
              const SizedBox(height: 18),
              Text(
                '이 지역에서 다르게 배출하는 품목',
                style: text.titleSmall?.copyWith(
                  color: Theme.of(context).colorScheme.primary,
                  fontWeight: FontWeight.w700,
                ),
              ),
              const SizedBox(height: 6),
              for (final entry in overrides.entries)
                _CategoryRule(
                  category: guide.categoryById(entry.key),
                  categoryId: entry.key,
                  rule: entry.value,
                ),
            ],
            if (region.sources.isNotEmpty) ...[
              const SizedBox(height: 18),
              Text(
                '출처',
                style: text.titleSmall?.copyWith(
                  color: Theme.of(context).colorScheme.primary,
                  fontWeight: FontWeight.w700,
                ),
              ),
              for (final s in region.sources)
                Padding(
                  padding: const EdgeInsets.only(top: 6),
                  child: InkWell(
                    onTap: () => launchUrl(
                      Uri.parse(s.url),
                      mode: LaunchMode.externalApplication,
                    ),
                    child: Row(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const Icon(
                          Icons.open_in_new_rounded,
                          size: 15,
                          color: Colors.white54,
                        ),
                        const SizedBox(width: 6),
                        Expanded(
                          child: Text(
                            s.title,
                            style: text.bodySmall?.copyWith(
                              color: Colors.lightBlueAccent,
                              decoration: TextDecoration.underline,
                              decorationColor: Colors.lightBlueAccent,
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
            ],
            const SizedBox(height: 18),
            Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Icon(
                  Icons.warning_amber_rounded,
                  size: 16,
                  color: Colors.amberAccent,
                ),
                const SizedBox(width: 6),
                Expanded(
                  child: Text(
                    '지자체 기준은 수시로 바뀝니다. 배출 요일·전용 용기·대형폐기물 신고 방법은 '
                    '반드시 거주지 시·군·구 공고로 최종 확인하세요.',
                    style: text.bodySmall?.copyWith(color: Colors.amberAccent),
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}

/// 지역에서 덮어쓴 품목 하나.
class _CategoryRule extends StatelessWidget {
  const _CategoryRule({
    required this.category,
    required this.categoryId,
    required this.rule,
  });

  final RecyclingCategory? category;
  final String categoryId;
  /// 이 지역에서 덮어쓴 항목(name/bin/instruction).
  final Map<String, dynamic> rule;

  @override
  Widget build(BuildContext context) {
    final text = Theme.of(context).textTheme;
    final base = category;
    final name = (rule['name'] as String?) ?? base?.name ?? categoryId;
    final bin = (rule['bin'] as String?) ?? base?.bin;
    final instruction =
        (rule['instruction'] as String?) ?? base?.instruction;
    final color = base == null ? Colors.white54 : Color(base.color);

    return Padding(
      padding: const EdgeInsets.only(top: 8),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Container(
            width: 8,
            height: 8,
            margin: const EdgeInsets.only(top: 6, right: 8),
            decoration: BoxDecoration(color: color, shape: BoxShape.circle),
          ),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  bin == null ? name : '$name → $bin',
                  style: text.bodyMedium?.copyWith(fontWeight: FontWeight.w700),
                ),
                if (instruction != null)
                  Text(
                    instruction,
                    style: text.bodySmall?.copyWith(color: Colors.white70),
                  ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
