import 'dart:ui' as ui;

import 'package:flutter/material.dart';


/// 실시간 모드에서 카메라 미리보기 위에 세그멘테이션 마스크만 그립니다.
///
/// 플러그인의 네이티브 오버레이는 박스와 마스크를 항상 함께 그리고 둘 중 하나만 끌 수 없어,
/// 박스를 빼려면 오버레이를 통째로 끄고 마스크를 여기서 그려야 합니다.
class LiveMaskOverlay extends StatelessWidget {
  const LiveMaskOverlay({super.key, required this.mask});

  final ui.Image mask;

  @override
  Widget build(BuildContext context) {
    return IgnorePointer(
      child: CustomPaint(
        size: Size.infinite,
        painter: LiveMaskPainter(mask),
      ),
    );
  }
}

class LiveMaskPainter extends CustomPainter {
  LiveMaskPainter(this.mask);

  final ui.Image mask;

  @override
  void paint(Canvas canvas, Size size) {
    // 마스크 격자는 카메라 프레임 전체에 대응하므로 미리보기 크기에 맞춰 늘린다.
    canvas.drawImageRect(
      mask,
      Rect.fromLTWH(0, 0, mask.width.toDouble(), mask.height.toDouble()),
      Offset.zero & size,
      // 마스크 이미지에 이미 투명도가 들어 있다(안쪽은 옅게, 외곽선은 불투명).
      Paint()..filterQuality = FilterQuality.low,
    );
  }

  @override
  bool shouldRepaint(LiveMaskPainter oldDelegate) =>
      !identical(oldDelegate.mask, mask);
}
