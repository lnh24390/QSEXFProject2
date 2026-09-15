import 'package:flutter/widgets.dart' show Locale;
import 'package:geocoding/geocoding.dart';
import 'package:geolocator/geolocator.dart';

import '../config/app_config.dart';
import '../data/recycling_guide.dart';

/// 지역 자동 판별 결과.
class RegionLookup {
  const RegionLookup({this.region, this.administrativeName, this.error});

  /// 찾은 지역. 위치는 얻었지만 맞는 지역 규칙이 없으면 null 입니다.
  final RecyclingRegion? region;

  /// GPS 로 얻은 행정구역 이름(예: "제주특별자치도 제주시"). 실패하면 null.
  final String? administrativeName;

  /// 실패 사유(권한 거부·위치 꺼짐·시간 초과 등). 성공하면 null.
  final String? error;

  bool get ok => error == null;
}

/// GPS 로 현재 행정구역을 확인해 [RecyclingGuide] 의 지역 규칙과 맞춥니다.
///
/// 위치 권한과 (주소 변환을 위한) 네트워크가 필요하므로 실패할 수 있습니다.
/// 실패하면 [RegionLookup.error] 에 사유가 담기고, 앱은 공통 규칙을 쓰거나
/// 사용자가 세부 옵션에서 지역을 직접 고르면 됩니다.
class RegionService {
  const RegionService._();

  static Future<RegionLookup> detect(RecyclingGuide guide) async {
    try {
      if (!await Geolocator.isLocationServiceEnabled()) {
        return const RegionLookup(error: '위치 서비스가 꺼져 있습니다');
      }
      var permission = await Geolocator.checkPermission();
      if (permission == LocationPermission.denied) {
        permission = await Geolocator.requestPermission();
      }
      if (permission == LocationPermission.denied ||
          permission == LocationPermission.deniedForever) {
        return const RegionLookup(error: '위치 권한이 없습니다');
      }

      // 시·도 판별에는 낮은 정확도로 충분하고, 그만큼 빠르고 배터리를 덜 씁니다.
      final position = await Geolocator.getCurrentPosition(
        locationSettings: LocationSettings(
          accuracy: LocationAccuracy.low,
          timeLimit: AppConfig.regionLookupTimeout,
        ),
      );

      // geocoding 5 부터는 Geocoding 인스턴스를 만들어 씁니다.
      final geocoding = Geocoding(locale: const Locale('ko', 'KR'));
      final places = await geocoding.placemarkFromCoordinates(
        position.latitude,
        position.longitude,
      );
      if (places.isEmpty) {
        return const RegionLookup(error: '주소를 찾지 못했습니다');
      }

      final p = places.first;
      final parts = <String>[
        p.administrativeArea ?? '',
        p.subAdministrativeArea ?? '',
        p.locality ?? '',
      ].where((e) => e.isNotEmpty).toList();
      final name = parts.join(' ');

      // 맞는 지역 규칙이 없으면 공통(전국 기준) 규칙을 쓴다
      return RegionLookup(
        region: guide.matchRegion(name) ?? guide.fallbackRegion,
        administrativeName: name.isEmpty ? null : name,
      );
    } catch (e) {
      return RegionLookup(error: '$e');
    }
  }
}
