#Requires -RunAsAdministrator
param(
    [string[]]$RemoteAddress = @('LocalSubnet'),
    [ValidateSet('Private', 'Public', 'Domain')]
    [string[]]$Profile = @('Private')
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$program = (Resolve-Path -LiteralPath (Join-Path $projectRoot 'NeonStrike.exe')).Path
$ruleName = 'NeonStrike-LAN-TCP-8765'
# Update our own rule only; preserve outbound policy and all other rules.
$settings = @{
    Direction = 'Inbound'
    Action = 'Allow'
    Protocol = 'TCP'
    LocalPort = 8765
    Program = $program
    Profile = $Profile
    RemoteAddress = $RemoteAddress
    Enabled = 'True'
    EdgeTraversalPolicy = 'Block'
}
if (Get-NetFirewallRule -Name $ruleName -ErrorAction SilentlyContinue) {
    Set-NetFirewallRule -Name $ruleName @settings
} else {
    New-NetFirewallRule -Name $ruleName -DisplayName 'NeonStrike game server TCP 8765' @settings
}
Write-Host "NeonStrike rule ready: $program / TCP 8765 / $($Profile -join ', ') / $($RemoteAddress -join ', ')"
Write-Host 'Other firewall rules and normal PC work are unchanged.'
