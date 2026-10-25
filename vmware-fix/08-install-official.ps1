#requires -RunAsAdministrator
$ErrorActionPreference = 'Continue'
$log = Join-Path $env:TEMP 'vmware-fix8.log'
function L([string]$m) { $line = "[{0}] {1}" -f (Get-Date -Format 'HH:mm:ss'), $m; $line | Out-File -FilePath $log -Append -Encoding utf8 }
Set-Content -Path $log -Value "=== install official VMware Workstation 26H1 ===" -Encoding utf8
L ("elevated = " + (New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator))

$exe = 'D:\VMW_installer\VMware-Workstation-Full-26H1-25388281.exe'
L ("installer exists: " + (Test-Path $exe))
$sig = Get-AuthenticodeSignature $exe
L ("signature: status=$($sig.Status) signer=$($sig.SignerCertificate.Subject)")
if ($sig.Status -ne 'Valid') { L "ABORT: signature not valid"; exit 1 }

# stop running VMware apps
Get-Process vmware,vmware-tray,vmware-vmx,vmnetcfg -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2

# remove the services we created manually so the installer can create its own
foreach ($s in 'VMware Authorization Service','VMnetDHCP','VMware NAT Service') {
    & sc.exe stop $s 2>&1 | Out-Null
    $o = & sc.exe delete $s 2>&1
    L ("delete service " + $s + " => " + (($o -join ' ') -replace "\s+", ' '))
}

# silent install; try several known argument forms until one is accepted
$attempts = @(
    @('/s','/v','/qn','EULAS_AGREED=1','AUTOSOFTWAREUPDATE=0','DATACOLLECTION=0'),
    @('/s','/v/qn','EULAS_AGREED=1'),
    @('/s','/v','"/qn EULAS_AGREED=1"')
)
foreach ($a in $attempts) {
    L ("starting installer with args: " + ($a -join ' '))
    $p = Start-Process -FilePath $exe -ArgumentList $a -Wait -PassThru
    L ("installer exit code = " + $p.ExitCode)
    if ($p.ExitCode -eq 0 -or $p.ExitCode -eq 3010) { break }
    if ($p.ExitCode -eq 1639) { L "1639 = invalid command line, retrying with next form"; continue }
    break
}

Start-Sleep -Seconds 5
L "--- HKLM InstallPath ---"
foreach ($k in 'HKLM:\SOFTWARE\VMware, Inc.\VMware Workstation','HKLM:\SOFTWARE\WOW6432Node\VMware, Inc.\VMware Workstation') {
    if (Test-Path $k) {
        $v = Get-ItemProperty $k
        L ("$k :: InstallPath=$($v.InstallPath) InstallPath64=$($v.InstallPath64)")
    } else { L "$k MISSING" }
}
L "--- vmx86 driver ---"
Get-Service | Where-Object { $_.Name -match 'vmx86|vmnet|VMware|VMnet' } | ForEach-Object { L ("svc {0} = {1} / {2}" -f $_.Name, $_.Status, $_.StartType) }
foreach ($f in 'C:\Windows\System32\drivers\vmx86.sys','C:\Windows\System32\vmx86.sys') { if (Test-Path $f) { L "found $f ($((Get-Item $f).Length) bytes)" } }
L "--- install dir ---"
Get-ChildItem 'C:\Program Files (x86)\VMware','C:\Program Files\VMware','D:\VMW' -ErrorAction SilentlyContinue -Directory | ForEach-Object { L ("dir " + $_.FullName) }
if (Test-Path 'C:\Program Files (x86)\VMware\VMware Workstation\x64\vmware-vmx.exe') { L "vmx present: yes" } else { L "vmx present in default path: no" }
L "=== done ==="
