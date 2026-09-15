import 'package:flutter/material.dart';

import '../services/model_catalog.dart';

/// 번들된 .tflite 모델 중 하나를 고르는 바텀시트. 선택한 asset 경로를 반환합니다.
///
/// [onOpenSettings] 가 있으면 하단에 "세부 옵션 설정" 버튼을 보여주고,
/// 누르면 이 시트를 닫은 뒤 콜백을 호출합니다(이때 반환값은 null).
Future<String?> showModelPickerSheet(
  BuildContext context, {
  required ModelCatalog catalog,
  required String currentModel,
  VoidCallback? onOpenSettings,
}) {
  return showModalBottomSheet<String>(
    context: context,
    showDragHandle: true,
    builder: (context) {
      final items = catalog.models.isEmpty
          ? <String>[currentModel]
          : catalog.models;
      return SafeArea(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const Padding(
              padding: EdgeInsets.fromLTRB(20, 0, 20, 8),
              child: Align(
                alignment: Alignment.centerLeft,
                child: Text('모델 선택',
                    style: TextStyle(
                        fontSize: 18, fontWeight: FontWeight.w700)),
              ),
            ),
            if (catalog.models.isEmpty)
              const Padding(
                padding: EdgeInsets.symmetric(horizontal: 20, vertical: 8),
                child: Text(
                  'assets/models 폴더에 .tflite 모델이 없습니다.\n'
                  'ml/scripts/export.py 로 변환한 모델을 넣고 다시 빌드하세요.',
                  style: TextStyle(color: Colors.white70),
                ),
              ),
            Flexible(
              // RadioGroup + RadioListTile 조합은 탭해도 선택 표시만 바뀌고 onChanged 가 오지 않아
              // 시트가 닫히지 않았다. 탭하면 바로 선택값을 돌려주도록 ListTile 로 직접 구성한다.
              child: ListView(
                shrinkWrap: true,
                children: [
                  for (final m in items)
                    ListTile(
                      leading: Icon(
                        m == currentModel
                            ? Icons.radio_button_checked
                            : Icons.radio_button_unchecked,
                        color: m == currentModel
                            ? Theme.of(context).colorScheme.primary
                            : Colors.white54,
                      ),
                      title: Text(ModelCatalog.displayName(m)),
                      subtitle: Text(m, style: const TextStyle(fontSize: 11)),
                      selected: m == currentModel,
                      onTap: () => Navigator.of(context).pop(m),
                    ),
                ],
              ),
            ),
            if (onOpenSettings != null)
              Padding(
                padding: const EdgeInsets.fromLTRB(16, 4, 16, 0),
                child: SizedBox(
                  width: double.infinity,
                  child: OutlinedButton.icon(
                    onPressed: () {
                      Navigator.of(context).pop();
                      onOpenSettings();
                    },
                    icon: const Icon(Icons.settings_rounded),
                    label: const Text('세부 옵션 설정'),
                  ),
                ),
              ),
            const SizedBox(height: 8),
          ],
        ),
      );
    },
  );
}
