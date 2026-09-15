import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:trash_sorter/data/recycling_guide.dart';

void main() {
  late RecyclingGuide guide;

  setUpAll(() {
    final raw = File('assets/config/recycling_guide.json').readAsStringSync();
    guide = RecyclingGuide.fromJsonString(raw);
  });

  test('클래스명 정규화', () {
    expect(RecyclingGuide.normalizeClassName('Plastic bag & wrapper'),
        'plastic_bag_wrapper');
    expect(RecyclingGuide.normalizeClassName('  Drink-Can '), 'drink_can');
    expect(RecyclingGuide.normalizeClassName('Rope & strings'), 'rope_strings');
  });

  test('명시 매핑: COCO bottle → 투명 페트병', () {
    expect(guide.resolve('bottle').id, 'pet');
    expect(guide.isExplicitlyMapped('bottle'), isTrue);
  });

  test('규칙 매핑: 모르는 클래스도 키워드로 분류', () {
    expect(guide.resolve('Crushed Plastic Bottle').id, 'pet');
    expect(guide.resolve('aluminium can small').id, 'can');
    expect(guide.resolve('foam tray').id, 'styrofoam');
  });

  test('완전히 모르는 클래스는 fallback(일반쓰레기)', () {
    expect(guide.resolve('zzz_unknown').id, guide.fallback.id);
    expect(guide.fallback.id, 'general');
  });

  test('비쓰레기 클래스는 ignore', () {
    expect(guide.isIgnored('person'), isTrue);
    expect(guide.isIgnored('Dining Table'), isTrue);
    expect(guide.isIgnored('bottle'), isFalse);
  });

  test('모든 카테고리에 안내 문구가 있다', () {
    for (final c in guide.categories) {
      expect(c.name, isNotEmpty);
      expect(c.bin, isNotEmpty);
      expect(c.instruction, isNotEmpty);
    }
  });
}
