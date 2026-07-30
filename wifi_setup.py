import paramiko
import sys

hostname = 'dpsi-lfr.local'
username = 'dpsi-lfr'
password = 'toor'
ssid = 'dkpa 5ghz'
wifi_pass = '9810813081'

print(f"Connecting to {hostname}...")
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
try:
    ssh.connect(hostname, username=username, password=password, timeout=10)
    
    # Check if nmcli exists
    stdin, stdout, stderr = ssh.exec_command('which nmcli')
    nmcli_path = stdout.read().decode().strip()
    
    if nmcli_path:
        print("Using nmcli to add profile...")
    if nmcli_path:
        print("Using nmcli to configure wifi...")
        cmds = [
            f"echo '{password}' | sudo -S nmcli connection delete 'DKPA 2.4gHz' || true",
            f"echo '{password}' | sudo -S nmcli dev wifi connect 'DKPA 2.4gHz' password '{wifi_pass}'"
        ]
        for c in cmds:
            stdin, stdout, stderr = ssh.exec_command(c)
            out = stdout.read().decode(errors='replace')
            err = stderr.read().decode(errors='replace')
            print("CMD:", c)
            print("STDOUT:", out.encode('ascii', 'replace').decode())
            print("STDERR:", err.encode('ascii', 'replace').decode())
    else:
        print("nmcli not found, trying wpa_supplicant...")
        cmd = f"echo '{password}' | sudo -S bash -c \"wpa_passphrase '{ssid}' '{wifi_pass}' >> /etc/wpa_supplicant/wpa_supplicant.conf && wpa_cli -i wlan0 reconfigure\""
        stdin, stdout, stderr = ssh.exec_command(cmd)
        print("STDOUT:", stdout.read().decode())
        print("STDERR:", stderr.read().decode())
except Exception as e:
    print(f"Failed to connect or configure: {e}")
finally:
    ssh.close()
