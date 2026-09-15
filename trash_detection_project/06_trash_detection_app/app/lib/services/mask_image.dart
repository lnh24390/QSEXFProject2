import 'dart:async';
import 'dart:typed_data';
import 'dart:ui' as ui;

import 'photo_analysis.dart';

/// 마스크를 그릴 RGBA 픽셀 버퍼와 크기.
class MaskPixels {
  const MaskPixels({
    required this.rgba,
    required this.width,
    required this.height,
  });

  /// 픽셀당 4바이트(R,G,B,A, 프리멀티플라이). 안쪽은 옅게, 가장자리는 불투명하며 바깥은 완전 투명입니다.
  final Uint8List rgba;
  final int width;
  final int height;
}

/// 인스턴스별 마스크를 하나의 RGBA 버퍼로 합칩니다.
///
/// 플러그인이 주는 마스크 격자는 letterbox 여백을 뺀 **원본 영역 전체**에 대응하므로,
/// 사진·미리보기 위에 그대로 늘려 그리면 위치가 맞습니다.
/// 겹치는 부분은 신뢰도가 높은 검출이 이깁니다(낮은 것부터 칠하고 높은 것으로 덮어씀).
///
/// 물체 **안쪽**은 [fillOpacity] 로 옅게 칠하고, **가장자리**([strokeWidth] 픽셀)는 불투명하게 칠해
/// 사각형 박스 대신 물체 모양을 따라가는 외곽선이 보이게 합니다.
/// 알파를 픽셀에 직접 넣으므로(프리멀티플라이) 그릴 때 별도 투명도 처리가 필요 없습니다.
MaskPixels? buildMaskPixels(
  List<PhotoDetection> detections, {
  required int Function(PhotoDetection detection) colorOf,
  required double threshold,
  required double fillOpacity,
  required int strokeWidth,
}) {
  // 마스크가 있는 검출만, 신뢰도 높은 순으로 모은다
  final withMask = detections
      .where(
        (d) =>
            (d.result.mask?.isNotEmpty ?? false) &&
            d.result.mask!.first.isNotEmpty,
      )
      .toList()
    ..sort((a, b) => b.confidence.compareTo(a.confidence));
  if (withMask.isEmpty) return null;

  // 캔버스 크기는 가장 신뢰도 높은 마스크를 기준으로 삼는다.
  // (모든 마스크는 같은 격자를 쓰지만, 다르면 좌표를 맞출 수 없어 건너뛴다)
  final base = withMask.first.result.mask!;
  final height = base.length;
  final width = base.first.length;
  final rgba = Uint8List(width * height * 4);

  final fillAlpha = (fillOpacity.clamp(0.0, 1.0) * 255).round();
  final stroke = strokeWidth < 1 ? 1 : strokeWidth;

  // 신뢰도 낮은 것부터 칠해 높은 것이 위에 오게 한다
  for (final d in withMask.reversed) {
    final mask = d.result.mask!;
    if (mask.length != height || mask.first.length != width) continue;

    final color = colorOf(d);
    final r = (color >> 16) & 0xFF;
    final g = (color >> 8) & 0xFF;
    final b = color & 0xFF;

    bool inside(int x, int y) {
      if (x < 0 || y < 0 || x >= width || y >= height) return false;
      final row = mask[y];
      return row.length == width && row[x] >= threshold;
    }

    /// 가장자리 판정: 안쪽 픽셀인데 [stroke] 칸 안에 바깥 픽셀이 있으면 외곽선.
    bool isEdge(int x, int y) {
      for (var d = 1; d <= stroke; d++) {
        if (!inside(x - d, y) ||
            !inside(x + d, y) ||
            !inside(x, y - d) ||
            !inside(x, y + d)) {
          return true;
        }
      }
      return false;
    }

    for (var y = 0; y < height; y++) {
      final row = mask[y];
      if (row.length != width) continue;
      final rowBase = y * width;
      for (var x = 0; x < width; x++) {
        if (row[x] < threshold) continue;
        final alpha = isEdge(x, y) ? 255 : fillAlpha;
        final i = (rowBase + x) * 4;
        // 프리멀티플라이: 알파를 색에 미리 곱해 둔다
        rgba[i] = (r * alpha) ~/ 255;
        rgba[i + 1] = (g * alpha) ~/ 255;
        rgba[i + 2] = (b * alpha) ~/ 255;
        rgba[i + 3] = alpha;
      }
    }
  }

  return MaskPixels(rgba: rgba, width: width, height: height);
}

/// [buildMaskPixels] 결과를 캔버스에 그릴 수 있는 이미지로 만듭니다. 마스크가 없으면 null.
Future<ui.Image?> buildMaskImage(
  List<PhotoDetection> detections, {
  required int Function(PhotoDetection detection) colorOf,
  required double threshold,
  required double fillOpacity,
  required int strokeWidth,
}) async {
  final pixels = buildMaskPixels(
    detections,
    colorOf: colorOf,
    threshold: threshold,
    fillOpacity: fillOpacity,
    strokeWidth: strokeWidth,
  );
  if (pixels == null) return null;

  final completer = Completer<ui.Image>();
  ui.decodeImageFromPixels(
    pixels.rgba,
    pixels.width,
    pixels.height,
    ui.PixelFormat.rgba8888,
    completer.complete,
  );
  return completer.future;
}
