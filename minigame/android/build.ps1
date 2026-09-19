param(
    [string]$ToolCache = (Join-Path $env:LOCALAPPDATA 'NeonDriftBuild')
)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$toolRoot = Join-Path $ToolCache 'tools/android-15'
$platformJar = Join-Path $ToolCache 'platform/android-35/android.jar'
$utf8 = New-Object System.Text.UTF8Encoding($false)

function Invoke-Checked([string]$Executable, [string[]]$Arguments) {
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed ($LASTEXITCODE): $Executable" }
}
function Get-VerifiedArchive([string]$Name, [string]$Hash, [string]$Destination, [string]$Sentinel) {
    if (Test-Path -LiteralPath $Sentinel) { return }
    New-Item -ItemType Directory -Path $ToolCache -Force | Out-Null
    $zipPath = Join-Path $ToolCache $Name
    if (-not (Test-Path -LiteralPath $zipPath)) {
        Write-Host "Downloading official Android tools: $Name"
        Invoke-WebRequest -UseBasicParsing -Uri "https://dl.google.com/android/repository/$Name" -OutFile $zipPath
    }
    if ((Get-FileHash -LiteralPath $zipPath -Algorithm SHA1).Hash -ne $Hash) {
        throw "Archive checksum mismatch: $zipPath"
    }
    Expand-Archive -LiteralPath $zipPath -DestinationPath $Destination -Force
}

$javaCommand = Get-Command java -ErrorAction Stop
$javaBin = Split-Path $javaCommand.Source
$javac = Join-Path $javaBin 'javac.exe'
$keytool = Join-Path $javaBin 'keytool.exe'
if (-not (Test-Path -LiteralPath $javac)) { throw 'A full JDK is required, not just a Java runtime.' }
Get-VerifiedArchive 'build-tools_r35_windows.zip' 'AF059BB67CF7786F45EE0DB85E2D24985DF1B4B6' (Join-Path $ToolCache 'tools') (Join-Path $toolRoot 'aapt2.exe')
Get-VerifiedArchive 'platform-35_r02.zip' '0BB560A90A7A2CBD0DD8348224D518B638FE7949' (Join-Path $ToolCache 'platform') $platformJar

# A new directory per build avoids deleting or modifying earlier outputs.
$runId = Get-Date -Format 'yyyyMMdd-HHmmss-fff'
# Android native tools on Windows require an ASCII staging path.
$buildRoot = Join-Path $ToolCache "build/$runId"
$webAssets = Join-Path $buildRoot 'assets/web'
$classDir = Join-Path $buildRoot 'classes'
$dexDir = Join-Path $buildRoot 'dex'
$outputDir = $PSScriptRoot
foreach ($dir in @($webAssets, $classDir, $dexDir, $outputDir)) {
    New-Item -ItemType Directory -Path $dir -Force | Out-Null
}

$html = [IO.File]::ReadAllText((Join-Path $projectRoot 'index.html'), $utf8)
$spriteSource = [regex]::Match($html, '<img id="player-sprite" src="([^"]+)"').Groups[1].Value
if (-not $spriteSource) { throw 'Player sprite reference is missing.' }
$files = @('index.html', 'styles.css', 'game-core.js', 'game.js')
foreach ($match in [regex]::Matches($html, '(?:src|href)="([^"#]+)"')) {
    $relative = [Uri]::UnescapeDataString(($match.Groups[1].Value -split '[?#]')[0])
    if ($relative -match '^(https?:|data:)') { throw "Unexpected external resource: $relative" }
    $files += $relative
}
foreach ($relative in ($files | Select-Object -Unique)) {
    $source = [IO.Path]::GetFullPath((Join-Path $projectRoot $relative))
    if (-not $source.StartsWith($projectRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Asset is outside the project: $relative"
    }
    if (-not (Test-Path -LiteralPath $source -PathType Leaf)) { throw "Missing game asset: $relative" }
    $packagedName = if ($relative -eq $spriteSource) { 'player-fighter.png' } else { $relative }
    $destination = Join-Path $webAssets $packagedName
    New-Item -ItemType Directory -Path (Split-Path $destination) -Force | Out-Null
    Copy-Item -LiteralPath $source -Destination $destination
}
$html = $html.Replace('</head>', '<link rel="stylesheet" href="android.css"></head>')
$html = $html.Replace('src="' + $spriteSource + '"', 'src="player-fighter.png"')
[IO.File]::WriteAllText((Join-Path $webAssets 'index.html'), $html, $utf8)
$cssPath = Join-Path $webAssets 'styles.css'
$css = [IO.File]::ReadAllText($cssPath, $utf8)
$css = [regex]::Replace($css, '@import\s+url\([^;]+\);', '')
[IO.File]::WriteAllText($cssPath, $css, $utf8)
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'android.css') -Destination (Join-Path $webAssets 'android.css')

Write-Host 'Compiling Android resources and Java...'
$aapt = Join-Path $toolRoot 'aapt2.exe'
$resources = Join-Path $buildRoot 'resources.zip'
$unsigned = Join-Path $buildRoot 'unsigned.apk'
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'res') -Destination (Join-Path $buildRoot 'res') -Recurse
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'src') -Destination (Join-Path $buildRoot 'src') -Recurse
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'AndroidManifest.xml') -Destination (Join-Path $buildRoot 'AndroidManifest.xml')
Invoke-Checked $aapt @('compile', '--dir', (Join-Path $buildRoot 'res'), '-o', $resources)
Invoke-Checked $aapt @('link', '-o', $unsigned, '-I', $platformJar, '--manifest', (Join-Path $buildRoot 'AndroidManifest.xml'), $resources)
$javaSources = @(Get-ChildItem -LiteralPath (Join-Path $buildRoot 'src') -Recurse -Filter '*.java' | ForEach-Object { $_.FullName })
Invoke-Checked $javac (@('--release', '8', '-encoding', 'UTF-8', '-classpath', $platformJar, '-d', $classDir) + $javaSources)
$classes = @(Get-ChildItem -LiteralPath $classDir -Recurse -Filter '*.class' | ForEach-Object { $_.FullName })
Invoke-Checked $javaCommand.Source (@('-cp', (Join-Path $toolRoot 'lib/d8.jar'), 'com.android.tools.r8.D8', '--lib', $platformJar, '--min-api', '26', '--output', $dexDir) + $classes)

Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem
$zip = [IO.Compression.ZipFile]::Open($unsigned, [IO.Compression.ZipArchiveMode]::Update)
try {
    # Explicit '/' names avoid Windows AAPT creating backslashes in asset entries.
    foreach ($asset in Get-ChildItem -LiteralPath $webAssets -File -Recurse) {
        $entryName = 'assets/web/' + $asset.FullName.Substring($webAssets.Length + 1).Replace('\', '/')
        [IO.Compression.ZipFileExtensions]::CreateEntryFromFile($zip, $asset.FullName, $entryName) | Out-Null
    }
    foreach ($dex in Get-ChildItem -LiteralPath $dexDir -Filter '*.dex') {
        [IO.Compression.ZipFileExtensions]::CreateEntryFromFile($zip, $dex.FullName, $dex.Name) | Out-Null
    }
} finally { $zip.Dispose() }

Write-Host 'Aligning and signing installable APK...'
$aligned = Join-Path $buildRoot 'aligned.apk'
Invoke-Checked (Join-Path $toolRoot 'zipalign.exe') @('-f', '-p', '4', $unsigned, $aligned)
$keystore = Join-Path $ToolCache 'neon-drift-debug.keystore'
if (-not (Test-Path -LiteralPath $keystore)) {
    Invoke-Checked $keytool @('-genkeypair', '-keystore', $keystore, '-storetype', 'JKS', '-alias', 'androiddebugkey', '-storepass', 'android', '-keypass', 'android', '-keyalg', 'RSA', '-keysize', '2048', '-validity', '10000', '-dname', 'CN=Neon Drift Development,O=Neon Drift,C=KR', '-noprompt')
}
$apk = Join-Path $outputDir 'Neondrift-2.0.1-debug.apk'
$signer = Join-Path $toolRoot 'lib/apksigner.jar'
Invoke-Checked $javaCommand.Source @('-jar', $signer, 'sign', '--ks', $keystore, '--ks-key-alias', 'androiddebugkey', '--ks-pass', 'pass:android', '--key-pass', 'pass:android', '--out', $apk, $aligned)
Invoke-Checked $javaCommand.Source @('-jar', $signer, 'verify', '--verbose', $apk)
Invoke-Checked (Join-Path $toolRoot 'zipalign.exe') @('-c', '-v', '4', $apk)
$hash = (Get-FileHash -LiteralPath $apk -Algorithm SHA256).Hash.ToLowerInvariant()
[IO.File]::WriteAllText((Join-Path $outputDir 'Neondrift-2.0.1-debug.apk.sha256'), "$hash  Neondrift-2.0.1-debug.apk`n", $utf8)
Write-Host "APK ready: $apk"
