import 'package:flutter/services.dart';

import '../config/app_config.dart';

/// 앱에 번들된 모델 목록을 AssetManifest 에서 읽어옵니다.
///
/// `assets/models/*.tflite` 를 넣고 `flutter pub get` 만 하면 자동으로 목록에 나타납니다.
class ModelCatalog {
  ModelCatalog._(this.models);

  /// 사용 가능한 모델의 asset 경로 목록 (정렬됨).
  final List<String> models;

  static Future<ModelCatalog> load() async {
    final manifest = await AssetManifest.loadFromAssetBundle(rootBundle);
    final tflite = manifest
        .listAssets()
        .where((p) => p.startsWith('assets/models/') && p.endsWith('.tflite'))
        .toList()
      ..sort();
    return ModelCatalog._(tflite);
  }

  /// 처음 로드할 모델. [preferred](마지막으로 쓴 모델) → 기본 모델 → 첫 번째 번들 모델 순으로,
  /// 번들에 실제로 있는 것을 고릅니다.
  /// 번들된 모델이 하나도 없으면 null 이며, 이때는 검출을 시작할 수 없습니다.
  String? initialModel({String? preferred}) {
    if (preferred != null && models.contains(preferred)) return preferred;
    if (models.contains(AppConfig.defaultModelAsset)) {
      return AppConfig.defaultModelAsset;
    }
    return models.isEmpty ? null : models.first;
  }

  static String displayName(String modelPath) =>
      modelPath.split('/').last.replaceAll('.tflite', '');
}
