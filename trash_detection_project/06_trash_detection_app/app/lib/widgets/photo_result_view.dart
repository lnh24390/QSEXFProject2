import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';

import '../services/photo_analysis.dart';

/// 촬영한 사진 위에 세그멘테이션 마스크와 검출 박스를 그려 보여줍니다.
///
/// [imageSize] 를 모르면(분석 전) 사진만 화면에 맞춰 보여줍니다.
class PhotoResultView extends StatelessWidget {
  const PhotoResultView({
    super.key,
    required this.imageBytes,
    this.imageSize,
    this.detections = const [],
    this.maskImage,
    this.showBoxes = true,
    required this.colorFor,
    required this.labelFor,
  });

  final Uint8List imageBytes;
  final Size? imageSize;
  final List<PhotoDetection> detections;

  /// 세그멘테이션 모델일 때만 있는 마스크 이미지. detect 모델이면 null.
  final ui.Image? maskImage;

  /// 박스와 라벨을 그릴지. 세그 모델에서 마스크만 보고 싶을 때 false.
  final bool showBoxes;
  final Color Function(PhotoDetection detection) colorFor;
  final String Function(PhotoDetection detection) labelFor;

  /// 화면 표시용 디코딩 폭 상한. 12MP 원본을 그대로 올리면 메모리를 많이 쓴다.
  static const int _maxDecodeWidth = 1440;

  @override
  Widget build(BuildContext context) {
    final size = imageSize;
    final image = Image.memory(
      imageBytes,
      fit: size == null ? BoxFit.contain : BoxFit.fill,
      gaplessPlayback: true,
      cacheWidth: size != null && size.width > _maxDecodeWidth
          ? _maxDecodeWidth
          : null,
    );

    return ColoredBox(
      color: Colors.black,
      child: size == null
          ? Center(child: image)
          : Center(
              child: AspectRatio(
                aspectRatio: size.width / size.height,
                child: Stack(
                  fit: StackFit.expand,
                  children: [
                    image,
                    CustomPaint(
                      painter: DetectionBoxPainter(
                        detections: detections,
                        maskImage: maskImage,
                        showBoxes: showBoxes,
                        colorFor: colorFor,
                        labelFor: labelFor,
                      ),
                    ),
                  ],
                ),
              ),
            ),
    );
  }
}

/// 마스크(있으면)와 normalizedBox(0~1) 기준 검출 박스·라벨을 그립니다.
class DetectionBoxPainter extends CustomPainter {
  DetectionBoxPainter({
    required this.detections,
    required this.colorFor,
    required this.labelFor,
    this.maskImage,
    this.showBoxes = true,
  });

  final List<PhotoDetection> detections;
  final ui.Image? maskImage;
  final bool showBoxes;
  final Color Function(PhotoDetection detection) colorFor;
  final String Function(PhotoDetection detection) labelFor;

  @override
  void paint(Canvas canvas, Size size) {
    // 마스크를 먼저 깔고 그 위에 박스를 그린다.
    // 마스크 격자는 사진 영역 전체에 대응하므로 화면 크기에 맞춰 늘린다.
    final mask = maskImage;
    if (mask != null) {
      canvas.drawImageRect(
        mask,
        Rect.fromLTWH(0, 0, mask.width.toDouble(), mask.height.toDouble()),
        Offset.zero & size,
        // 마스크 이미지에 이미 투명도가 들어 있다(안쪽은 옅게, 외곽선은 불투명).
        Paint()..filterQuality = FilterQuality.low,
      );
    }

    if (!showBoxes) return;

    // 신뢰도 낮은 것부터 그려 높은 것이 위에 오게 한다
    for (final d in detections.reversed) {
      final nb = d.normalizedBox;
      final rect = Rect.fromLTRB(
        nb.left.clamp(0.0, 1.0) * size.width,
        nb.top.clamp(0.0, 1.0) * size.height,
        nb.right.clamp(0.0, 1.0) * size.width,
        nb.bottom.clamp(0.0, 1.0) * size.height,
      );
      if (rect.isEmpty) continue;
      final color = colorFor(d);

      canvas.drawRect(
        rect,
        Paint()
          ..color = color
          ..style = PaintingStyle.stroke
          ..strokeWidth = 3,
      );

      final text = TextPainter(
        text: TextSpan(
          text: labelFor(d),
          style: const TextStyle(
            color: Colors.white,
            fontSize: 13,
            fontWeight: FontWeight.w700,
          ),
        ),
        textDirection: TextDirection.ltr,
        maxLines: 1,
        ellipsis: '…',
      )..layout(maxWidth: size.width);

      const padH = 6.0;
      const padV = 3.0;
      final labelHeight = text.height + padV * 2;
      // 박스 위에 공간이 없으면 박스 안쪽 위에 붙인다
      final top = rect.top - labelHeight >= 0 ? rect.top - labelHeight : rect.top;
      final width = text.width + padH * 2;
      final left = (rect.left + width > size.width)
          ? (size.width - width).clamp(0.0, size.width)
          : rect.left;
      canvas.drawRect(
        Rect.fromLTWH(left, top, width, labelHeight),
        Paint()..color = color,
      );
      text.paint(canvas, Offset(left + padH, top + padV));
    }
  }

  @override
  bool shouldRepaint(DetectionBoxPainter oldDelegate) =>
      !identical(oldDelegate.detections, detections) ||
      !identical(oldDelegate.maskImage, maskImage) ||
      oldDelegate.showBoxes != showBoxes;
}
