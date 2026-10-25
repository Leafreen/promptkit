#requires -RunAsAdministrator
$ErrorActionPreference = 'Continue'
$log = Join-Path $env:TEMP 'vmware-fix5.log'
function L([string]$m) { $line = "[{0}] {1}" -f (Get-Date -Format 'HH:mm:ss'), $m; $line | Out-File -FilePath $log -Append -Encoding utf8 }
Set-Content -Path $log -Value "=== create vmnet adapters (setupapi) ===" -Encoding utf8
L ("elevated = " + (New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator))

$cs = @'
using System;
using System.Runtime.InteropServices;

public static class DevInst {
    [StructLayout(LayoutKind.Sequential)]
    public struct SP_DEVINFO_DATA {
        public int cbSize;
        public Guid ClassGuid;
        public int DevInst;
        public IntPtr Reserved;
    }
    const int DIF_REGISTERDEVICE = 0x00000019;
    const int SPDRP_HARDWAREID = 0x00000001;
    const int INSTALLFLAG_FORCE = 0x00000001;

    [DllImport("setupapi.dll", CharSet = CharSet.Auto, SetLastError = true)]
    static extern IntPtr SetupDiCreateDeviceInfoList(ref Guid ClassGuid, IntPtr hwndParent);
    [DllImport("setupapi.dll", CharSet = CharSet.Auto, SetLastError = true)]
    static extern bool SetupDiCreateDeviceInfo(IntPtr DeviceInfoSet, string DeviceName, ref Guid ClassGuid,
        string DeviceDescription, IntPtr hwndParent, int CreationFlags, ref SP_DEVINFO_DATA DeviceInfoData);
    [DllImport("setupapi.dll", CharSet = CharSet.Auto, SetLastError = true)]
    static extern bool SetupDiSetDeviceRegistryProperty(IntPtr DeviceInfoSet, ref SP_DEVINFO_DATA DeviceInfoData,
        int Property, byte[] PropertyBuffer, int PropertyBufferSize);
    [DllImport("setupapi.dll", SetLastError = true)]
    static extern bool SetupDiCallClassInstaller(int InstallFunction, IntPtr DeviceInfoSet, ref SP_DEVINFO_DATA DeviceInfoData);
    [DllImport("setupapi.dll", SetLastError = true)]
    static extern bool SetupDiDestroyDeviceInfoList(IntPtr DeviceInfoSet);
    [DllImport("newdev.dll", CharSet = CharSet.Auto, SetLastError = true)]
    static extern bool UpdateDriverForPlugAndPlayDevices(IntPtr hwndParent, string HardwareId,
        string FullInfPath, int InstallFlags, out bool bRebootRequired);

    static Guid NetClass = new Guid("4d36e972-e325-11ce-bfc1-08002be10318");

    public static string Install(string instanceName, string hardwareId, string infPath) {
        IntPtr set = SetupDiCreateDeviceInfoList(ref NetClass, IntPtr.Zero);
        if (set == IntPtr.Zero || set == (IntPtr)(-1)) return "SetupDiCreateDeviceInfoList failed: " + Marshal.GetLastWin32Error();
        SP_DEVINFO_DATA did = new SP_DEVINFO_DATA();
        did.cbSize = Marshal.SizeOf(typeof(SP_DEVINFO_DATA));
        if (!SetupDiCreateDeviceInfo(set, instanceName, ref NetClass, instanceName, IntPtr.Zero, 0x00000001, ref did))
            return "SetupDiCreateDeviceInfo failed: " + Marshal.GetLastWin32Error();

        byte[] id = System.Text.Encoding.Unicode.GetBytes(hardwareId + "\0\0");
        if (!SetupDiSetDeviceRegistryProperty(set, ref did, SPDRP_HARDWAREID, id, id.Length))
            return "SetupDiSetDeviceRegistryProperty failed: " + Marshal.GetLastWin32Error();

        if (!SetupDiCallClassInstaller(DIF_REGISTERDEVICE, set, ref did))
            return "DIF_REGISTERDEVICE failed: " + Marshal.GetLastWin32Error();

        bool reboot;
        bool ok = UpdateDriverForPlugAndPlayDevices(IntPtr.Zero, hardwareId, infPath, INSTALLFLAG_FORCE, out reboot);
        int last = Marshal.GetLastWin32Error();
        SetupDiDestroyDeviceInfoList(set);
        if (!ok) return "UpdateDriverForPlugAndPlayDevices failed: " + last;
        return "OK rebootRequired=" + reboot;
    }
}
'@
Add-Type -TypeDefinition $cs -Language CSharp

$inf = 'C:\Windows\INF\oem48.inf'
if (-not (Test-Path $inf)) { $inf = 'D:\VMW\netadapter.inf' }
L "INF = $inf"

foreach ($n in 8, 1) {
    $hwid = "*VMnetAdapter$n"
    try {
        $r = [DevInst]::Install("VMnetAdapter$n", $hwid, $inf)
        L "install $hwid => $r"
    } catch { L "install $hwid => EXCEPTION $($_.Exception.Message)" }
}

Start-Sleep -Seconds 3
L "--- drivers ---"
Get-Service | Where-Object { $_.Name -match 'vmnet|VMware|VMnet' } | ForEach-Object { L ("svc {0} = {1} / {2}" -f $_.Name, $_.Status, $_.StartType) }
L "--- adapters ---"
Get-NetAdapter -ErrorAction SilentlyContinue | ForEach-Object { L ("nic {0} | {1} | {2}" -f $_.Name, $_.InterfaceDescription, $_.Status) }
L "=== done ==="
