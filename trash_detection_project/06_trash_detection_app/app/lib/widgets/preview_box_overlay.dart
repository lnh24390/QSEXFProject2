import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';

import '../config/app_config.dart';

/// 사진 모드 미리보기 위에 **검은 박스만** 그립니다(이름·점수 없음).
///
/// 찍기 전에는 "무엇이 잡혔는지" 를 알려주지 않고 **어디에 초점을 맞출지**만 보여줍니다.
/// 품목 이름과 색은 촬영 후 결과(PhotoResultView)에서 처음 나옵니다.
/// 플러그인 네이티브 오버레이는 항상 클래스명과 점수를 함께 그리고 라벨만 끌 수 없어,
/// 사진 모드에서는 오버레이를 끄고 여기서 직접 그립니다.
///
/// 프레임마다 다시 그려야 하므로 setState 대신 [boxes] 를 듣습니다.
class PreviewBoxOverlay extends StatelessWidget {
  const PreviewBoxOverlay({super.key, required this.boxes});

  /// 0~1 로 정규화된 박스 목록. 새 프레임이 오면 값만 바뀝니다.
  final ValueListenable<List<Rect>> boxes;

  @override
  Widget build(BuildContext context) {
    return IgnorePointer(
      child: RepaintBoundary(
        child: CustomPaint(
          size: Size.infinite,
          painter: _PreviewBoxPainter(boxes),
        ),
      ),
    );
  }
}

class _PreviewBoxPainter extends CustomPainter {
  _PreviewBoxPainter(this.boxes) : super(repaint: boxes);

  final ValueListenable<List<Rect>> boxes;

  @override
  void paint(Canvas canvas, Size size) {
    final list = boxes.value;
    if (list.isEmpty) return;

    // 어두운 배경에서도 테두리가 보이도록 흰 테두리를 먼저 깔고 그 위에 검은 선을 얹는다
    final outline = Paint()
      ..color = Colors.white.withValues(alpha: 0.55)
      ..style = PaintingStyle.stroke
      ..strokeWidth = AppConfig.previewBoxStrokeWidth + 2;
    final stroke = Paint()
      ..color = Colors.black
      ..style = PaintingStyle.stroke
      ..strokeWidth = AppConfig.previewBoxStrokeWidth;

    for (final nb in list) {
      final rect = Rect.fromLTRB(
        nb.left.clamp(0.0, 1.0) * size.width,
        nb.top.clamp(0.0, 1.0) * size.height,
        nb.right.clamp(0.0, 1.0) * size.width,
        nb.bottom.clamp(0.0, 1.0) * size.height,
      );
      if (rect.isEmpty) continue;
      final rrect = RRect.fromRectAndRadius(rect, const Radius.circular(4));
      canvas.drawRRect(rrect, outline);
      canvas.drawRRect(rrect, stroke);
    }
  }

  // repaint 로 boxes 를 듣고 있어 값이 바뀌면 알아서 다시 그린다
  @override
  bool shouldRepaint(_PreviewBoxPainter oldDelegate) => false;
}
