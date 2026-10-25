$ErrorActionPreference = 'Continue'
$log = Join-Path $env:TEMP 'vmware-fix4.log'
Set-Content -Path $log -Value "=== restart VNE elevated ===" -Encoding utf8
& taskkill.exe /IM vmnetcfg.exe /F 2>&1 | Out-File $log -Append -Encoding utf8
Start-Sleep -Seconds 2
Start-Process -FilePath 'D:\VMW\vmnetcfg.exe'
Start-Sleep -Seconds 5
Get-Process vmnetcfg -ErrorAction SilentlyContinue | Select-Object Id,StartTime | Out-File $log -Append -Encoding utf8
