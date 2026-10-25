#requires -RunAsAdministrator
$ErrorActionPreference = 'Continue'
$log = Join-Path $env:TEMP 'vmware-fix3.log'
function L([string]$m) { $line = "[{0}] {1}" -f (Get-Date -Format 'HH:mm:ss'), $m; $line | Out-File -FilePath $log -Append -Encoding utf8 }
Set-Content -Path $log -Value "=== VMware repair 3: vmnet components ===" -Encoding utf8
L ("elevated = " + (New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator))

$vnetlib = 'D:\VMW\vnetlib.exe'
function Run([string[]]$cli) {
    $args2 = @('--') + $cli
    $o = & $vnetlib @args2 2>&1
    $code = $LASTEXITCODE
    $txt = (($o | Out-String) -replace "\s+", ' ').Trim()
    L ("vnetlib -- " + ($cli -join ' ') + " => exit=$code out=[$txt]")
}

# install bridge (NDIS protocol), NAT service, DHCP service, then adapters
Run @('install','bridge')
Run @('install','nat')
Run @('install','dhcp')
foreach ($v in 'vmnet8','vmnet1') {
    Run @('install','adapter',$v)
    Run @('install','adapter',($v -replace 'vmnet',''))
}

L "--- services ---"
Get-Service | Where-Object { $_.Name -match 'vmnet|VMware|VMnet' } | ForEach-Object { L ("svc {0} = {1} / {2}" -f $_.Name, $_.Status, $_.StartType) }
L "--- adapters ---"
Get-NetAdapter -ErrorAction SilentlyContinue | ForEach-Object { L ("nic {0} | {1} | {2}" -f $_.Name, $_.InterfaceDescription, $_.Status) }
L "--- vmnetlib registry ---"
$base = 'HKLM:\SOFTWARE\VMware, Inc.\VMnetLib'
if (Test-Path $base) {
    Get-ChildItem $base -Recurse -ErrorAction SilentlyContinue | ForEach-Object {
        $k = $_
        $vals = (Get-ItemProperty $k.PSPath -ErrorAction SilentlyContinue).PSObject.Properties | Where-Object { $_.Name -notlike 'PS*' } | ForEach-Object { "$($_.Name)=$($_.Value)" }
        L ("reg " + ($k.Name -replace '.*VMnetLib','VMnetLib') + " :: " + ($vals -join '; '))
    }
} else { L "no VMnetLib key" }
L "=== done ==="
