$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$compiler = 'C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe'
$output = Join-Path $projectRoot 'NeonStrike.exe'
$source = Join-Path $PSScriptRoot 'NeonStrikeLauncher.cs'
$index = Join-Path $projectRoot 'dist\index.html'
$styles = Join-Path $projectRoot 'dist\styles.css'
$game = Join-Path $projectRoot 'dist\game.js'
$icon = Join-Path $PSScriptRoot 'NeonStrike.ico'
$iconBuilder = Join-Path $PSScriptRoot 'create-icon.ps1'

& $iconBuilder | Out-Null
& $compiler /nologo /target:winexe /optimize+ /win32icon:$icon /out:$output /reference:System.Windows.Forms.dll /resource:$index,index.html /resource:$styles,styles.css /resource:$game,game.js /resource:$icon,NeonStrike.ico $source
if ($LASTEXITCODE -ne 0) { throw "C# compiler failed with exit code $LASTEXITCODE" }
Write-Output "Built $output"
