$ErrorActionPreference = 'Stop'
# Run in an elevated PowerShell window. Private LAN access only.
$ruleName = 'NeonStrike-LAN-TCP-8765'
if (-not (Get-NetFirewallRule -Name $ruleName -ErrorAction SilentlyContinue)) {
    New-NetFirewallRule -Name $ruleName -DisplayName 'NeonStrike LAN multiplayer' -Direction Inbound -Action Allow -Protocol TCP -LocalPort 8765 -Profile Private -RemoteAddress LocalSubnet
}
Write-Host 'NeonStrike TCP 8765 private LAN rule is ready.'
