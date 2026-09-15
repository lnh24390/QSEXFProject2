import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:trash_sorter/widgets/preview_box_overlay.dart';

void main() {
  testWidgets('미리보기 박스에는 어떤 글자도 표시하지 않는다', (tester) async {
    final boxes = ValueNotifier<List<Rect>>([
      const Rect.fromLTRB(0.1, 0.1, 0.5, 0.6),
      const Rect.fromLTRB(0.6, 0.2, 0.9, 0.8),
    ]);
    addTearDown(boxes.dispose);

    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(
          body: SizedBox(
            width: 400,
            height: 300,
            child: PreviewBoxOverlay(boxes: boxes),
          ),
        ),
      ),
    );

    // 찍기 전에는 무엇이 잡혔는지(클래스명·점수) 알려주지 않는다
    expect(find.byType(Text), findsNothing);
    expect(find.byType(RichText), findsNothing);
    expect(find.byType(CustomPaint), findsWidgets);
  });

  testWidgets('박스가 없거나 값이 바뀌어도 예외 없이 그린다', (tester) async {
    final boxes = ValueNotifier<List<Rect>>(const []);
    addTearDown(boxes.dispose);

    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(
          body: SizedBox(
            width: 200,
            height: 200,
            child: PreviewBoxOverlay(boxes: boxes),
          ),
        ),
      ),
    );
    expect(tester.takeException(), isNull);

    // 화면 밖으로 넘치는 값이 와도 잘라서 그린다
    boxes.value = [const Rect.fromLTRB(-0.5, -0.2, 1.4, 1.9)];
    await tester.pump();
    expect(tester.takeException(), isNull);

    // 빈 박스(넓이 0)는 건너뛴다
    boxes.value = [const Rect.fromLTRB(0.3, 0.3, 0.3, 0.3)];
    await tester.pump();
    expect(tester.takeException(), isNull);
  });
}
