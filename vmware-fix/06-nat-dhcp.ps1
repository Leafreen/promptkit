#requires -RunAsAdministrator
$ErrorActionPreference = 'Continue'
$log = Join-Path $env:TEMP 'vmware-fix6.log'
function L([string]$m) { $line = "[{0}] {1}" -f (Get-Date -Format 'HH:mm:ss'), $m; $line | Out-File -FilePath $log -Append -Encoding utf8 }
Set-Content -Path $log -Value "=== vmnet drivers + NAT/DHCP config ===" -Encoding utf8

# 1) VMnetuserif driver (vmnat + vmnetdhcp depend on it)
L "--- install VMnetuserif ---"
$o = & rundll32.exe setupapi.dll,InstallHinfSection DefaultInstall 132 D:\VMW\netuserif.inf 2>&1
L ("rundll32 netuserif => " + (($o | Out-String) -replace "\s+", ' '))
Start-Sleep -Seconds 2
$svc = Get-Service -Name 'VMnetuserif' -ErrorAction SilentlyContinue
L ("VMnetuserif service: " + $(if ($svc) { "$($svc.Status)/$($svc.StartType)" } else { 'MISSING' }))

# 2) vmnet registry config
$vnetBase = 'HKLM:\SOFTWARE\VMware, Inc.\VMnetLib\VMnetConfig'
$cfg = @{
    'vmnet8' = @{ subnet = '192.168.226.0'; mask = '255.255.255.0'; adapter = '192.168.226.1'; nat = 1; dhcp = 1 }
    'vmnet1' = @{ subnet = '192.168.19.0';  mask = '255.255.255.0'; adapter = '192.168.19.1';  nat = 0; dhcp = 1 }
}
foreach ($k in $cfg.Keys) {
    $path = Join-Path $vnetBase $k
    New-Item -Path $path -Force | Out-Null
    $v = $cfg[$k]
    New-ItemProperty -Path $path -Name 'DisplayName'      -Value $k              -PropertyType String -Force | Out-Null
    New-ItemProperty -Path $path -Name 'IPSubnetAddress'  -Value $v.subnet       -PropertyType String -Force | Out-Null
    New-ItemProperty -Path $path -Name 'IPSubnetMask'     -Value $v.mask         -PropertyType String -Force | Out-Null
    New-ItemProperty -Path $path -Name 'AdapterIPAddress' -Value $v.adapter      -PropertyType String -Force | Out-Null
    New-ItemProperty -Path $path -Name 'UseAdapter'       -Value 1               -PropertyType DWord  -Force | Out-Null
    New-ItemProperty -Path $path -Name 'NAT'              -Value $v.nat          -PropertyType DWord  -Force | Out-Null
    New-ItemProperty -Path $path -Name 'DHCP'             -Value $v.dhcp         -PropertyType DWord  -Force | Out-Null
    L "registry $k configured"
}

# 3) config files for vmnat / vmnetdhcp
$pd = 'C:\ProgramData\VMware'
New-Item -ItemType Directory -Path $pd -Force | Out-Null
$natConf = @"
[host]
ip = 192.168.226.1
device = vmnet8
activeFTP = 1
dnsProxy = 1

[incomingtcp]

[incomingudp]

"@
Set-Content -Path (Join-Path $pd 'vmnetnat.conf') -Value $natConf -Encoding ascii
$dhcpConf = @"
# VMware VMnet DHCP configuration
allow unknown-clients;
default-lease-time 1800;
max-lease-time 7200;

subnet 192.168.226.0 netmask 255.255.255.0 {
	range 192.168.226.128 192.168.226.254;
	option broadcast-address 192.168.226.255;
	option domain-name-servers 192.168.226.2;
	option domain-name "localdomain";
	option routers 192.168.226.2;
}

subnet 192.168.19.0 netmask 255.255.255.0 {
	range 192.168.19.128 192.168.19.254;
	option broadcast-address 192.168.19.255;
	option domain-name-servers 192.168.19.2;
	option domain-name "localdomain";
	option routers 192.168.19.2;
}
"@
Set-Content -Path (Join-Path $pd 'vmnetdhcp.conf') -Value $dhcpConf -Encoding ascii
L "conf files written"

# 4) static IPs on the two vmnet adapters
$map = @{ 'VMware Virtual Ethernet Adapter for VMnet8' = '192.168.226.1'; 'VMware Virtual Ethernet Adapter for VMnet1' = '192.168.19.1' }
foreach ($desc in $map.Keys) {
    $nic = Get-NetAdapter | Where-Object { $_.InterfaceDescription -eq $desc } | Select-Object -First 1
    if (-not $nic) { L "adapter not found: $desc"; continue }
    $alias = $nic.Name
    & netsh.exe interface ipv4 set address name="$alias" static $($map[$desc]) 255.255.255.0 2>&1 | Out-Null
    L ("set " + $alias + " -> " + $map[$desc])
}

# 5) start NAT + DHCP services
foreach ($s in 'VMnetuserif','VMnetDHCP','VMware NAT Service') {
    $o = & sc.exe start $s 2>&1
    $st = (Get-Service -Name $s -ErrorAction SilentlyContinue).Status
    L ("start " + $s + " => " + $st)
}

Start-Sleep -Seconds 2
L "--- final state ---"
Get-Service | Where-Object { $_.Name -match 'vmnet|VMware|VMnet' } | ForEach-Object { L ("svc {0} = {1} / {2}" -f $_.Name, $_.Status, $_.StartType) }
Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.InterfaceAlias -match 'VMnet' -or $_.InterfaceDescription -match 'VMware' } | ForEach-Object { L ("ip " + $_.InterfaceAlias + " " + $_.IPAddress + "/" + $_.PrefixLength) }
L "=== done ==="
