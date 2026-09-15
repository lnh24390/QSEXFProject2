import 'dart:convert';

import 'package:flutter/services.dart';

/// 쓰레기 "상태"에 따른 배출 방법 한 줄.
///
/// 모델은 오염·코팅 여부를 판별하지 못하므로, 사용자가 눈으로 확인할 기준을 안내합니다.
/// 예) when: "코팅되어 있거나 영수증(감열지)이면", then: "general" → "일반쓰레기로"
class GuideCondition {
  const GuideCondition({required this.when, this.thenCategoryId, this.note});

  /// 상태 설명. "~이면/~있으면" 형태.
  final String when;

  /// 그 상태일 때 대신 배출할 카테고리 id. 없으면 분류는 그대로고 [note] 만 안내합니다.
  final String? thenCategoryId;

  /// 추가 안내(제거·헹굼 등). 없을 수 있습니다.
  final String? note;

  factory GuideCondition.fromJson(Map<String, dynamic> json) {
    return GuideCondition(
      when: json['when'] as String? ?? '',
      thenCategoryId: json['then'] as String?,
      note: json['note'] as String?,
    );
  }
}

/// 지역별 분리수거 규칙. 공통 규칙 위에 일부 카테고리만 덮어씁니다.
class RegionSource {
  const RegionSource({required this.title, required this.url});

  final String title;
  final String url;

  factory RegionSource.fromJson(Map<String, dynamic> json) => RegionSource(
        title: json['title'] as String? ?? '',
        url: json['url'] as String? ?? '',
      );
}

class RecyclingRegion {
  const RecyclingRegion({
    required this.id,
    required this.name,
    required this.matches,
    this.notice,
    this.summary,
    this.updated,
    this.sources = const [],
    this.categoryOverrides = const {},
  });

  final String id;
  final String name;

  /// GPS 로 얻은 행정구역 이름에 이 문자열이 하나라도 들어 있으면 이 지역으로 봅니다.
  final List<String> matches;

  /// 지역 전체에 해당하는 한 줄 안내(안내 패널에 표시). 없을 수 있습니다.
  final String? notice;

  /// 지역 안내 화면에 보여줄 자세한 설명. 없을 수 있습니다.
  final String? summary;

  /// 내용을 확인한 시점(예: "2026-09-15 확인"). 지자체 기준은 바뀌므로 함께 보여줍니다.
  final String? updated;

  /// 근거 자료 링크.
  final List<RegionSource> sources;

  /// 카테고리 id → 덮어쓸 항목(name/bin/instruction/conditions).
  final Map<String, Map<String, dynamic>> categoryOverrides;

  factory RecyclingRegion.fromJson(String id, Map<String, dynamic> json) {
    final overrides = <String, Map<String, dynamic>>{};
    (json['categories'] as Map<String, dynamic>? ?? {}).forEach((k, v) {
      if (v is Map<String, dynamic>) overrides[k] = v;
    });
    return RecyclingRegion(
      id: id,
      name: json['name'] as String? ?? id,
      matches: (json['match'] as List<dynamic>? ?? [])
          .map((e) => e.toString())
          .toList(),
      notice: json['notice'] as String?,
      summary: json['summary'] as String?,
      updated: json['updated'] as String?,
      sources: (json['sources'] as List<dynamic>? ?? [])
          .whereType<Map<String, dynamic>>()
          .map(RegionSource.fromJson)
          .where((s) => s.title.isNotEmpty)
          .toList(),
      categoryOverrides: overrides,
    );
  }
}

/// 분리수거 카테고리(버릴 곳) 정보.
class RecyclingCategory {
  const RecyclingCategory({
    required this.id,
    required this.name,
    required this.bin,
    required this.instruction,
    required this.color,
    required this.icon,
    this.conditions = const [],
  });

  final String id;
  final String name;
  final String bin;
  final String instruction;
  final int color;
  final String icon;

  /// 상태에 따라 다르게 버려야 하는 경우들. JSON 의 categories[*].conditions.
  final List<GuideCondition> conditions;

  factory RecyclingCategory.fromJson(String id, Map<String, dynamic> json) {
    return RecyclingCategory(
      id: id,
      name: json['name'] as String,
      bin: json['bin'] as String,
      instruction: json['instruction'] as String,
      color: _parseHexColor(json['color'] as String? ?? '#546E7A'),
      icon: json['icon'] as String? ?? id,
      conditions: (json['conditions'] as List<dynamic>? ?? [])
          .whereType<Map<String, dynamic>>()
          .map(GuideCondition.fromJson)
          .where((c) => c.when.isNotEmpty)
          .toList(),
    );
  }

  /// 지역 규칙으로 일부 항목만 바꾼 사본을 만듭니다. 없는 항목은 그대로 둡니다.
  RecyclingCategory applyOverride(Map<String, dynamic> override) {
    if (override.isEmpty) return this;
    return RecyclingCategory(
      id: id,
      name: override['name'] as String? ?? name,
      bin: override['bin'] as String? ?? bin,
      instruction: override['instruction'] as String? ?? instruction,
      color: color,
      icon: icon,
      conditions: override.containsKey('conditions')
          ? (override['conditions'] as List<dynamic>? ?? [])
                .whereType<Map<String, dynamic>>()
                .map(GuideCondition.fromJson)
                .where((c) => c.when.isNotEmpty)
                .toList()
          : conditions,
    );
  }

  static int _parseHexColor(String hex) {
    final cleaned = hex.replaceFirst('#', '');
    final value = int.parse(cleaned, radix: 16);
    return cleaned.length == 6 ? (0xFF000000 | value) : value;
  }
}

class _KeywordRule {
  const _KeywordRule(this.keywords, this.categoryId);
  final List<String> keywords;
  final String categoryId;
}

/// `assets/config/recycling_guide.json` 을 읽어 모델 클래스명 → 분리수거 카테고리로 변환합니다.
///
/// 모델을 바꿔도(클래스명이 달라져도) JSON만 수정하면 되도록 앱 코드와 분리되어 있습니다.
class RecyclingGuide {
  RecyclingGuide._({
    required this._categories,
    required this._classMap,
    required this._rules,
    required this._ignore,
    required this._fallbackId,
    required this.regions,
  });

  static const String assetPath = 'assets/config/recycling_guide.json';

  final Map<String, RecyclingCategory> _categories;
  final Map<String, String> _classMap;
  final List<_KeywordRule> _rules;
  final Set<String> _ignore;
  final String _fallbackId;

  /// JSON 에 적힌 순서대로의 지역 목록. 위에 있는 지역부터 매칭합니다.
  final List<RecyclingRegion> regions;
  final Map<String, RecyclingCategory> _cache = {};

  static Future<RecyclingGuide> load() async {
    final raw = await rootBundle.loadString(assetPath);
    return RecyclingGuide.fromJsonString(raw);
  }

  static RecyclingGuide fromJsonString(String raw) {
    final json = jsonDecode(raw) as Map<String, dynamic>;

    final categories = <String, RecyclingCategory>{};
    (json['categories'] as Map<String, dynamic>).forEach((id, value) {
      categories[id] =
          RecyclingCategory.fromJson(id, value as Map<String, dynamic>);
    });

    final classMap = <String, String>{};
    (json['classes'] as Map<String, dynamic>? ?? {}).forEach((k, v) {
      classMap[normalizeClassName(k)] = v as String;
    });

    final rules = <_KeywordRule>[];
    for (final r in (json['rules'] as List<dynamic>? ?? [])) {
      final m = r as Map<String, dynamic>;
      rules.add(_KeywordRule(
        (m['contains'] as List<dynamic>).map((e) => e.toString()).toList(),
        m['category'] as String,
      ));
    }

    final ignore = (json['ignore'] as List<dynamic>? ?? [])
        .map((e) => normalizeClassName(e.toString()))
        .toSet();

    final fallback = json['fallback'] as String? ?? 'general';
    if (!categories.containsKey(fallback)) {
      throw StateError('fallback 카테고리 "$fallback" 가 categories에 없습니다.');
    }

    final regions = <RecyclingRegion>[];
    (json['regions'] as Map<String, dynamic>? ?? {}).forEach((id, value) {
      if (value is Map<String, dynamic>) {
        regions.add(RecyclingRegion.fromJson(id, value));
      }
    });

    return RecyclingGuide._(
      categories: categories,
      classMap: classMap,
      rules: rules,
      ignore: ignore,
      fallbackId: fallback,
      regions: regions,
    );
  }

  /// 모델 라벨 정규화: 소문자, 공백/하이픈/&/슬래시 → '_', 연속 '_' 축약.
  static String normalizeClassName(String name) {
    return name
        .trim()
        .toLowerCase()
        .replaceAll(RegExp(r'[\s\-&/]+'), '_')
        .replaceAll(RegExp(r'_+'), '_')
        .replaceAll(RegExp(r'^_|_$'), '');
  }

  RecyclingCategory get fallback => _categories[_fallbackId]!;

  /// 카테고리 id 로 직접 찾기. 조건(then)의 카테고리 이름을 보여줄 때 씁니다.
  RecyclingCategory? categoryById(String id) => _categories[id];

  /// id 로 지역 찾기. 없으면 null.
  RecyclingRegion? regionById(String? id) {
    if (id == null) return null;
    for (final r in regions) {
      if (r.id == id) return r;
    }
    return null;
  }

  /// match 가 비어 있는 지역(= 공통 규칙). 어느 지역에도 맞지 않을 때 씁니다.
  RecyclingRegion? get fallbackRegion {
    for (final r in regions) {
      if (r.matches.isEmpty) return r;
    }
    return null;
  }

  /// GPS 로 얻은 행정구역 문자열(예: "제주특별자치도 제주시")에 맞는 지역을 찾습니다.
  ///
  /// 위에 적힌 지역부터 검사하고, 맞는 것이 없으면 null 을 돌려줍니다.
  RecyclingRegion? matchRegion(String administrativeName) {
    if (administrativeName.isEmpty) return null;
    for (final r in regions) {
      for (final keyword in r.matches) {
        if (keyword.isNotEmpty && administrativeName.contains(keyword)) {
          return r;
        }
      }
    }
    return null;
  }

  /// 클래스명을 지역 규칙까지 반영해 카테고리로 바꿉니다.
  ///
  /// [regionId] 가 없거나 그 지역에 해당 카테고리 규칙이 없으면 공통 규칙을 그대로 씁니다.
  RecyclingCategory resolveFor(String className, {String? regionId}) {
    final base = resolve(className);
    final region = regionById(regionId);
    final override = region?.categoryOverrides[base.id];
    return override == null ? base : base.applyOverride(override);
  }

  Iterable<RecyclingCategory> get categories => _categories.values;

  /// 클래스명이 명시적으로 매핑되어 있는지 여부 (규칙/기본값 제외).
  bool isExplicitlyMapped(String className) =>
      _classMap.containsKey(normalizeClassName(className));

  /// 쓰레기가 아닌 클래스(사람·동물·차량 등)인지 여부. true 면 검출 결과에서 제외합니다.
  bool isIgnored(String className) =>
      _ignore.contains(normalizeClassName(className));

  RecyclingCategory resolve(String className) {
    final key = normalizeClassName(className);
    final cached = _cache[key];
    if (cached != null) return cached;

    RecyclingCategory? result;
    final mappedId = _classMap[key];
    if (mappedId != null) result = _categories[mappedId];

    if (result == null) {
      for (final rule in _rules) {
        if (rule.keywords.every(key.contains)) {
          result = _categories[rule.categoryId];
          if (result != null) break;
        }
      }
    }

    result ??= fallback;
    _cache[key] = result;
    return result;
  }
}
