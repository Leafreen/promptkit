$ErrorActionPreference = 'Continue'
$src = 'D:\ubuntu'
$dst = 'D:\ubuntu_test'
$vmxName = 'Ubuntu 64-bit.vmx'
$orig = Get-Content (Join-Path $src $vmxName) -Encoding Default

$candidates = @(
    'monitorType = "2"',
    'monitorType = "3"',
    'monitorType = "0"',
    'monitorModeUseULM = "TRUE"',
    'monitorModeIsULM = "TRUE"',
    'monitorMode = "ulm"'
)

foreach ($c in $candidates) {
    if (Test-Path $dst) { Remove-Item $dst -Recurse -Force -ErrorAction SilentlyContinue }
    New-Item -ItemType Directory $dst | Out-Null
    Copy-Item "$src\*" $dst -Recurse -Force -ErrorAction SilentlyContinue
    Remove-Item "$dst\*.lck" -Recurse -Force -ErrorAction SilentlyContinue
    $lines = $orig + $c
    Set-Content -Path (Join-Path $dst $vmxName) -Value $lines -Encoding UTF8

    Stop-Process -Name vmware,vmware-tray,vmware-vmx -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 2
    Start-Process -FilePath 'D:\VMW\vmware.exe' -ArgumentList '-X',("`"$dst\$vmxName`"") | Out-Null
    Start-Sleep -Seconds 16

    $log = Get-ChildItem "$env:TEMP\vmware-DELL\vmware-vmx-*.log" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    $hit = @()
    if ($log) {
        $hit = (Select-String -Path $log.FullName -Pattern 'IOPL_Init|ULM:|WinHv|WHP|openFailed|initFailed|Cannot find VMX|Failed to initialize' | Select-Object -Last 4 | ForEach-Object { ($_.Line -replace '^.*threadName="vmx"\] ', '') })
    }
    "==== $c"
    if ($hit.Count -eq 0) { "   (no vmx log yet)" } else { $hit | ForEach-Object { "   $_" } }
}
Stop-Process -Name vmware,vmware-tray -Force -ErrorAction SilentlyContinue

