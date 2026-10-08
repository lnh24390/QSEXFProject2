param([string]$ToolCache = (Join-Path (Split-Path (Split-Path $PSScriptRoot)) 'android-build-cache'),[string]$JavaBin = 'C:\Program Files\JetBrains\PyCharm 2026.1\jbr\bin')
$ErrorActionPreference='Stop'
function Run([string]$exe,[string[]]$arguments){ & $exe @arguments; if($LASTEXITCODE -ne 0){throw "Build failed: $exe"} }
$tools=Join-Path $ToolCache 'tools\android-15'
$platform=Join-Path $ToolCache 'platform\android-35\android.jar'
$run=Join-Path $ToolCache ('build\'+(Get-Date -Format 'yyyyMMdd-HHmmss'))
$classes=Join-Path $run 'classes';$dex=Join-Path $run 'dex'
New-Item -ItemType Directory -Force $classes,$dex | Out-Null
$resources=Join-Path $run 'resources.zip';$unsigned=Join-Path $run 'unsigned.apk'
Run (Join-Path $tools 'aapt2.exe') @('compile','--dir',(Join-Path $PSScriptRoot 'res'),'-o',$resources)
Run (Join-Path $tools 'aapt2.exe') @('link','-o',$unsigned,'-I',$platform,'--manifest',(Join-Path $PSScriptRoot 'AndroidManifest.xml'),$resources)
$sources=@(Get-ChildItem (Join-Path $PSScriptRoot 'src') -Recurse -Filter '*.java' | ForEach-Object FullName)
Run (Join-Path $JavaBin 'javac.exe') (@('--release','8','-encoding','UTF-8','-classpath',$platform,'-d',$classes)+$sources)
$classFiles=@(Get-ChildItem $classes -Recurse -Filter '*.class' | ForEach-Object FullName)
Run (Join-Path $JavaBin 'java.exe') (@('-cp',(Join-Path $tools 'lib\d8.jar'),'com.android.tools.r8.D8','--lib',$platform,'--min-api','26','--output',$dex)+$classFiles)
Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem
$zip=[IO.Compression.ZipFile]::Open($unsigned,[IO.Compression.ZipArchiveMode]::Update)
try {foreach($file in Get-ChildItem $dex -Filter '*.dex'){[IO.Compression.ZipFileExtensions]::CreateEntryFromFile($zip,$file.FullName,$file.Name)|Out-Null}}finally{$zip.Dispose()}
$aligned=Join-Path $run 'aligned.apk'
Run (Join-Path $tools 'zipalign.exe') @('-f','-p','4',$unsigned,$aligned)
$key=Join-Path $ToolCache 'neonstrike-debug.keystore'
if(-not(Test-Path $key)){Run (Join-Path $JavaBin 'keytool.exe') @('-genkeypair','-keystore',$key,'-storetype','JKS','-alias','androiddebugkey','-storepass','android','-keypass','android','-keyalg','RSA','-keysize','2048','-validity','10000','-dname','CN=Neon Strike Development,O=Neon Strike,C=KR','-noprompt')}
$apk=Join-Path $PSScriptRoot 'NeonStrike-1.0.5-debug.apk';$signer=Join-Path $tools 'lib\apksigner.jar'
Run (Join-Path $JavaBin 'java.exe') @('-jar',$signer,'sign','--ks',$key,'--ks-key-alias','androiddebugkey','--ks-pass','pass:android','--key-pass','pass:android','--out',$apk,$aligned)
Run (Join-Path $JavaBin 'java.exe') @('-jar',$signer,'verify','--verbose',$apk)
Run (Join-Path $tools 'zipalign.exe') @('-c','4',$apk)
(Get-FileHash $apk -Algorithm SHA256).Hash | Set-Content (Join-Path $PSScriptRoot 'NeonStrike-1.0.5-debug.apk.sha256')
Write-Output "APK ready: $apk"
