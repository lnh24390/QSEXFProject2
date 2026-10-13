$ErrorActionPreference = 'Stop'
$executable = Join-Path $PSScriptRoot 'NeonStrike.exe'
if (-not (Test-Path -LiteralPath $executable)) { throw 'NeonStrike.exe was not found.' }
Start-Process -FilePath $executable -ArgumentList '--host-only' -WindowStyle Hidden
Write-Host 'Starting NeonStrike in the background without a game window.'
Write-Host 'Clients connect to http://<server-PC-IP>:8765/ . You can keep working on this PC.'
