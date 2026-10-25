#requires -RunAsAdministrator
$ErrorActionPreference = 'Continue'
$log = Join-Path $env:TEMP 'vmware-fix2.log'
function L([string]$m) { $line = "[{0}] {1}" -f (Get-Date -Format 'HH:mm:ss'), $m; $line | Out-File -FilePath $log -Append -Encoding utf8 }
Set-Content -Path $log -Value "=== VMware repair 2: WHPX + vmnet driver staging ===" -Encoding utf8
L ("elevated = " + (New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator))

# 1) Windows Hypervisor Platform (needed because VBS/Hyper-V owns VT-x and no vmx86.sys ships with WS26)
L "--- DISM: enable HypervisorPlatform ---"
$dism = & dism.exe /Online /Enable-Feature /FeatureName:HypervisorPlatform /All /NoRestart 2>&1
L ($dism -join "`r`n")
$feat = Get-WindowsOptionalFeature -Online -FeatureName HypervisorPlatform -ErrorAction SilentlyContinue
L ("HypervisorPlatform state = " + $feat.State)

# 2) Stage vmnet driver packages so adapters can be created later
L "--- pnputil: add driver packages ---"
foreach ($inf in 'netuserif.inf','netbridge.inf','netadapter.inf') {
    $path = Join-Path 'D:\VMW' $inf
    if (-not (Test-Path $path)) { L "missing $path"; continue }
    $o = & pnputil.exe /add-driver $path /install 2>&1
    L ("pnputil $inf => " + (($o -join ' | ') -replace "\s+", ' '))
}

# 3) Report services / adapters state
L "--- services ---"
Get-Service | Where-Object { $_.Name -match 'vmnet|VMware|VMnet|vmci|vmusb' } | ForEach-Object { L ("svc {0} = {1} / {2}" -f $_.Name, $_.Status, $_.StartType) }
L "--- adapters ---"
Get-NetAdapter -ErrorAction SilentlyContinue | ForEach-Object { L ("nic {0} | {1} | {2}" -f $_.Name, $_.InterfaceDescription, $_.Status) }
L "=== done ==="
