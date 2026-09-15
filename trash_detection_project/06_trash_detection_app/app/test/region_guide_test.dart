import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:trash_sorter/data/recycling_guide.dart';

/// 지역 규칙 덮어쓰기 동작 확인용 최소 JSON.
const _json = '''
{
  "fallback": "general",
  "categories": {
    "paper": {
      "name": "종이", "bin": "종이류 수거함", "instruction": "펼쳐서 묶어 배출",
      "color": "#F4511E", "icon": "paper",
      "conditions": [{"when": "코팅되어 있으면", "then": "general"}]
    },
    "general": {
      "name": "일반쓰레기", "bin": "종량제 봉투", "instruction": "종량제 봉투에 배출",
      "color": "#546E7A", "icon": "general"
    }
  },
  "classes": {"paper": "paper"},
  "regions": {
    "jeju": {
      "name": "제주특별자치도",
      "match": ["제주"],
      "notice": "요일별 배출제",
      "categories": {
        "paper": {"bin": "클린하우스 종이류", "instruction": "지정 요일에 배출"}
      }
    },
    "seoul": {"name": "서울특별시", "match": ["서울"], "categories": {}}
  }
}
''';

void main() {
  late RecyclingGuide sample;
  late RecyclingGuide bundled;

  setUpAll(() {
    sample = RecyclingGuide.fromJsonString(_json);
    bundled = RecyclingGuide.fromJsonString(
      File('assets/config/recycling_guide.json').readAsStringSync(),
    );
  });

  test('행정구역 이름으로 지역을 찾는다', () {
    expect(sample.matchRegion('제주특별자치도 제주시')?.id, 'jeju');
    expect(sample.matchRegion('서울특별시 강남구')?.id, 'seoul');
    expect(sample.matchRegion('부산광역시 해운대구'), isNull);
    expect(sample.matchRegion(''), isNull);
  });

  test('지역 규칙이 있으면 해당 항목만 덮어쓴다', () {
    final common = sample.resolveFor('paper');
    expect(common.bin, '종이류 수거함');

    final jeju = sample.resolveFor('paper', regionId: 'jeju');
    expect(jeju.bin, '클린하우스 종이류');
    expect(jeju.instruction, '지정 요일에 배출');
    // 덮어쓰지 않은 항목은 공통 값을 유지한다
    expect(jeju.name, common.name);
    expect(jeju.color, common.color);
    expect(jeju.conditions.length, common.conditions.length);
  });

  test('match 가 비어 있는 지역이 공통 규칙이 된다', () {
    // 최소 JSON 에는 공통 지역이 없다
    expect(sample.fallbackRegion, isNull);
    // 실제 안내 파일에는 default 지역이 공통 규칙
    expect(bundled.fallbackRegion?.id, 'default');
    expect(bundled.fallbackRegion?.notice, isNotNull);
  });

  test('지역에 규칙이 없거나 모르는 지역이면 공통 규칙을 쓴다', () {
    expect(sample.resolveFor('paper', regionId: 'seoul').bin, '종이류 수거함');
    expect(sample.resolveFor('paper', regionId: 'unknown').bin, '종이류 수거함');
  });

  test('지역 안내에 설명·출처·확인 시점이 들어 있다 (화면에서 읽을 수 있어야 함)', () {
    for (final id in ['jeju', 'seoul', 'default']) {
      final r = bundled.regionById(id)!;
      expect(r.summary, isNotNull, reason: '$id 설명 없음');
      expect(r.updated, isNotNull, reason: '$id 확인 시점 없음');
      expect(r.sources, isNotEmpty, reason: '$id 출처 없음');
      for (final src in r.sources) {
        expect(src.url, startsWith('http'), reason: '$id 출처 링크 형식');
      }
    }
  });

  test('제주는 품목별 배출 요일이 규칙으로 들어 있다', () {
    final jeju = bundled.regionById('jeju')!;
    expect(jeju.categoryOverrides.keys, contains('plastic'));
    expect(jeju.categoryOverrides.keys, contains('paper'));
    // 공통 규칙과 실제로 달라야 의미가 있다
    final common = bundled.resolveFor('paper');
    final local = bundled.resolveFor('paper', regionId: 'jeju');
    expect(local.instruction, isNot(common.instruction));
    expect(local.instruction, contains('요일'));
  });

  test('실제 안내 파일에도 지역과 상태 조건이 들어 있다', () {
    expect(bundled.regions.map((r) => r.id), contains('jeju'));
    expect(bundled.regionById('jeju')?.notice, isNotNull);
    // 모든 카테고리에 상태별 안내가 최소 1개
    for (final c in bundled.categories) {
      expect(c.conditions, isNotEmpty, reason: '${c.id} 에 conditions 없음');
    }
  });
}
