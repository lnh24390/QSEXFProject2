/// 검출 화면의 동작 모드.
enum DetectionMode {
  /// 카메라 프레임을 계속 분석해 실시간으로 안내.
  live('실시간'),

  /// 촬영한 사진 한 장을 분석해 안내.
  photo('사진');

  const DetectionMode(this.label);

  /// UI 표시용 이름.
  final String label;
}
