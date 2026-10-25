$ErrorActionPreference = 'Continue'
$log = Join-Path $env:TEMP 'vmware-fix.log'
function L([string]$m) { $line = "[{0}] {1}" -f (Get-Date -Format 'HH:mm:ss'), $m; $line | Out-File -FilePath $log -Append -Encoding utf8; }
Set-Content -Path $log -Value "=== VMware repair: registry + authd service ===" -Encoding utf8
L ("elevated = " + (New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator))

$installDir = 'D:\VMW\'
$installDir64 = 'D:\VMW\x64\'

foreach ($base in @('HKLM:\SOFTWARE\VMware, Inc.', 'HKLM:\SOFTWARE\WOW6432Node\VMware, Inc.')) {
    $key = Join-Path $base 'VMware Workstation'
    try {
        New-Item -Path $key -Force -ErrorAction Stop | Out-Null
        New-ItemProperty -Path $key -Name 'InstallPath'        -Value $installDir   -PropertyType String -Force -ErrorAction Stop | Out-Null
        New-ItemProperty -Path $key -Name 'InstallPath64'      -Value $installDir64 -PropertyType String -Force -ErrorAction Stop | Out-Null
        New-ItemProperty -Path $key -Name 'ProductVersion'     -Value '26.0.0'      -PropertyType String -Force -ErrorAction Stop | Out-Null
        New-ItemProperty -Path $key -Name 'ProductBuildNumber' -Value '25388281'    -PropertyType String -Force -ErrorAction Stop | Out-Null
        L "OK   $key"
    } catch {
        L "FAIL $key : $($_.Exception.Message)"
    }
    try {
        $p = Get-ItemProperty -Path $key -ErrorAction Stop
        L ("     InstallPath=" + $p.InstallPath + " InstallPath64=" + $p.InstallPath64)
    } catch { L "     (read-back failed)" }
}

# --- VMware Authorization Service (vmware-authd.exe) ---
$svcName = 'VMware Authorization Service'
$existing = Get-Service -Name $svcName -ErrorAction SilentlyContinue
if ($existing) {
    L "authd service already exists: $($existing.Status)"
} else {
    $out = & sc.exe create $svcName binPath= 'D:\VMW\vmware-authd.exe' start= auto DisplayName= 'VMware Authorization Service' 2>&1
    L ("sc create authd: " + ($out -join ' | '))
}
$out = & sc.exe start $svcName 2>&1
L ("sc start authd: " + ($out -join ' | '))
L ("authd status: " + (Get-Service -Name $svcName -ErrorAction SilentlyContinue).Status)

L "=== done ==="
