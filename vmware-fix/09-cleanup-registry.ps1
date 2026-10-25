#requires -RunAsAdministrator
$ErrorActionPreference = 'Continue'
$log = Join-Path $env:TEMP 'vmware-fix9.log'
function L([string]$m) { $line = "[{0}] {1}" -f (Get-Date -Format 'HH:mm:ss'), $m; $line | Out-File -FilePath $log -Append -Encoding utf8 }
Set-Content -Path $log -Value "=== cleanup leftover manual registrations ===" -Encoding utf8

# remove the fake 32-bit view key we created earlier (points at the broken D:\VMW repack)
foreach ($k in 'HKLM:\SOFTWARE\WOW6432Node\VMware, Inc.\VMware Workstation') {
    if (Test-Path $k) {
        try { Remove-Item $k -Recurse -Force -ErrorAction Stop; L "deleted $k" } catch { L "FAILED delete $k : $($_.Exception.Message)" }
    } else { L "not present: $k" }
}
# remove fake HKCU key
$hk = 'HKCU:\SOFTWARE\VMware, Inc.\VMware Workstation'
if (Test-Path $hk) {
    try { Remove-Item $hk -Recurse -Force -ErrorAction Stop; L "deleted $hk" } catch { L "FAILED delete $hk : $($_.Exception.Message)" }
} else { L "not present: $hk" }

# report authoritative InstallPath
foreach ($k in 'HKLM:\SOFTWARE\VMware, Inc.\VMware Workstation','HKLM:\SOFTWARE\WOW6432Node\VMware, Inc.\VMware Workstation') {
    if (Test-Path $k) { $v = Get-ItemProperty $k; L "$k :: InstallPath=$($v.InstallPath) InstallPath64=$($v.InstallPath64)" } else { L "$k absent" }
}
L ("vmx86.sys: " + (Test-Path 'C:\Windows\System32\drivers\vmx86.sys'))
L ("vmx exe: " + (Test-Path 'C:\Program Files\VMware\VMware Workstation\x64\vmware-vmx.exe'))
L "=== done ==="
