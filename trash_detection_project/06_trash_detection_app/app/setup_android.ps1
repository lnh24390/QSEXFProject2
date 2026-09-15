# Flutter Android 프로젝트 파일 생성 + 플러그인 요구사항 패치
# 사용: app 폴더에서  powershell -ExecutionPolicy Bypass -File .\setup_android.ps1
#
# 하는 일
#  1. flutter create --platforms=android .   (이미 있는 lib/, pubspec.yaml, AndroidManifest.xml 은 덮어쓰지 않음)
#  2. android/app/build.gradle.kts 에 minSdk 23 이상 / release proguard 설정 보강
#  3. flutter pub get

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Get-Command flutter -ErrorAction SilentlyContinue)) {
    Write-Error "flutter 명령을 찾을 수 없습니다. Flutter SDK(3.47 이상 stable)를 설치하고 PATH 에 추가하세요."
}

flutter --version

# 1) 플랫폼 파일 생성 (기존 파일은 보존)
flutter create --platforms=android --org com.example --project-name trash_sorter .

# 2) build.gradle.kts 패치
$gradle = Join-Path $PSScriptRoot "android\app\build.gradle.kts"
if (-not (Test-Path $gradle)) { Write-Error "build.gradle.kts 를 찾지 못했습니다: $gradle" }
$content = Get-Content $gradle -Raw

# minSdk: flutter.minSdkVersion(기본 24) 이 23 이상이므로 그대로 두되, 숫자로 고정된 경우 23 미만이면 올림
if ($content -match 'minSdk\s*=\s*(\d+)') {
    if ([int]$Matches[1] -lt 23) {
        $content = $content -replace 'minSdk\s*=\s*\d+', 'minSdk = 23'
        Write-Host "minSdk -> 23 으로 수정"
    }
}

# Kotlin Android 플러그인 적용 (MainActivity.kt 가 Kotlin 이라 필수).
# settings.gradle.kts 에는 apply false 로 선언만 되어 있어, 모듈에서 적용하지 않으면
# AGP 9 에서 kotlin { compilerOptions { ... } } 블록이 Unresolved reference 로 깨진다.
if ($content -notmatch 'org\.jetbrains\.kotlin\.android') {
    $content = $content -replace '(id\("com\.android\.application"\))', "`$1`n    id(`"org.jetbrains.kotlin.android`")"
    Write-Host "Kotlin Android 플러그인 적용 추가"
}

# release 빌드 proguard 규칙 추가 (LiteRT 클래스 보존)
if ($content -notmatch 'proguard-rules.pro') {
    $content = $content -replace '(release\s*\{)', "`$1`n            isMinifyEnabled = true`n            isShrinkResources = true`n            proguardFiles(getDefaultProguardFile(`"proguard-android-optimize.txt`"), `"proguard-rules.pro`")"
    Write-Host "release proguard 설정 추가"
}

Set-Content -Path $gradle -Value $content -Encoding utf8

# 3) 의존성
flutter pub get

Write-Host ""
Write-Host "완료. 다음 단계:"
Write-Host "  1) ml/scripts/export.py 로 만든 .tflite 를 app/assets/models/ 에 넣기 (없으면 yolo26n 공식 모델을 첫 실행 시 다운로드)"
Write-Host "  2) 기기 연결 후:  flutter run --release"
