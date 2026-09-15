import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:trash_sorter/data/recycling_guide.dart';
import 'package:trash_sorter/services/detection_aggregator.dart';
import 'package:trash_sorter/widgets/guide_panel.dart';

void main() {
  late RecyclingGuide guide;

  setUpAll(() {
    guide = RecyclingGuide.fromJsonString(
      File('assets/config/recycling_guide.json').readAsStringSync(),
    );
  });

  Widget wrap(Widget child) => MaterialApp(home: Scaffold(body: child));

  testWidgets('검출이 없으면 촬영 안내 문구를 보여준다', (tester) async {
    await tester.pumpWidget(wrap(
      GuidePanel(guide: guide, detections: const [], maxItems: 3),
    ));
    expect(find.text('분리수거할 쓰레기를 화면에 담아주세요'), findsOneWidget);
  });

  testWidgets('모델 로딩 중에는 로딩 문구를 보여준다', (tester) async {
    await tester.pumpWidget(wrap(
      GuidePanel(
          guide: guide, detections: const [], maxItems: 3, modelLoading: true),
    ));
    expect(find.textContaining('모델을 불러오는 중'), findsOneWidget);
  });

  testWidgets('검출된 품목의 버릴 곳과 방법을 보여준다', (tester) async {
    final det = ActiveDetection(
      className: 'bottle',
      confidence: 0.91,
      lastSeen: DateTime.now(),
    );
    await tester.pumpWidget(wrap(
      GuidePanel(guide: guide, detections: [det], maxItems: 3),
    ));
    final pet = guide.resolve('bottle');
    expect(find.text('${pet.name} → ${pet.bin}'), findsOneWidget);
    expect(find.text(pet.instruction), findsOneWidget);
    expect(find.text('bottle 91%'), findsOneWidget);
  });

  testWidgets('사진에서 못 찾으면 각도·거리를 바꿔 다시 찍도록 안내한다', (tester) async {
    await tester.pumpWidget(wrap(
      GuidePanel(
        guide: guide,
        detections: const [],
        maxItems: 3,
        emptyPrompt: GuidePrompt.photoNothing,
      ),
    ));

    expect(GuidePrompt.photoNothing.subtitle, contains('각도'));
    expect(GuidePrompt.photoNothing.subtitle, contains('거리'));
    expect(find.text(GuidePrompt.photoNothing.subtitle), findsOneWidget);

    // 조치 목록이 빠짐없이 화면에 있어야 한다
    expect(GuidePrompt.photoNothing.tips, isNotEmpty);
    for (final tip in GuidePrompt.photoNothing.tips) {
      expect(find.text(tip), findsOneWidget, reason: tip);
    }

    // 촬영 전 문구에는 조치 목록을 붙이지 않는다
    expect(GuidePrompt.photoReady.tips, isEmpty);
  });

  testWidgets('사진 모드: 촬영 전·분석 중·결과 없음 문구를 보여준다', (tester) async {
    await tester.pumpWidget(wrap(
      GuidePanel(
        guide: guide,
        detections: const [],
        maxItems: 3,
        emptyPrompt: GuidePrompt.photoReady,
      ),
    ));
    expect(find.text(GuidePrompt.photoReady.title), findsOneWidget);

    await tester.pumpWidget(wrap(
      GuidePanel(
        guide: guide,
        detections: const [],
        maxItems: 3,
        modelLoading: true,
        busyPrompt: GuidePrompt.photoAnalyzing,
        emptyPrompt: GuidePrompt.photoReady,
      ),
    ));
    await tester.pumpAndSettle();
    expect(find.text(GuidePrompt.photoAnalyzing.title), findsOneWidget);

    await tester.pumpWidget(wrap(
      GuidePanel(
        guide: guide,
        detections: const [],
        maxItems: 3,
        emptyPrompt: GuidePrompt.photoNothing,
      ),
    ));
    await tester.pumpAndSettle();
    expect(find.text(GuidePrompt.photoNothing.title), findsOneWidget);
  });

  testWidgets('분류가 다른 후보가 있으면 "또는" 안내를 함께 보여준다', (tester) async {
    final det = ActiveDetection(
      className: 'paper',
      confidence: 0.70,
      lastSeen: DateTime.now(),
      alternativeClassName: 'vinyl',
      alternativeConfidence: 0.62,
    );
    await tester.pumpWidget(wrap(
      GuidePanel(guide: guide, detections: [det], maxItems: 3),
    ));
    final vinyl = guide.resolve('vinyl');
    expect(
      find.text('또는 ${vinyl.name} → ${vinyl.bin} 일 수 있어요 (vinyl 62%)'),
      findsOneWidget,
    );
  });

  testWidgets('첫 품목에는 상태별 배출 안내를 함께 보여준다', (tester) async {
    final det = ActiveDetection(
      className: 'paper',
      confidence: 0.9,
      lastSeen: DateTime.now(),
    );
    await tester.pumpWidget(
      wrap(GuidePanel(guide: guide, detections: [det], maxItems: 3)),
    );
    final paper = guide.resolve('paper');
    final first = paper.conditions.first;
    final target = guide.categoryById(first.thenCategoryId!)!;
    // "코팅되어 있거나 영수증(감열지)이면 → 일반쓰레기" 형태
    expect(find.text('${first.when} → ${target.name}'), findsOneWidget);
  });

  testWidgets('maxItems 를 넘는 품목은 표시하지 않는다', (tester) async {
    final now = DateTime.now();
    final dets = [
      for (final c in ['bottle', 'can', 'book', 'cup'])
        ActiveDetection(className: c, confidence: 0.8, lastSeen: now),
    ];
    await tester.pumpWidget(wrap(
      GuidePanel(guide: guide, detections: dets, maxItems: 2),
    ));
    expect(find.textContaining('%'), findsNWidgets(2));
  });
}
