import os
import paramiko

print("Stopping motors...")
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(os.environ.get('PI_IP', '192.168.1.12'), username='dpsi-lfr', password='toor')

script = '''
import pigpio
import time
pi = pigpio.pi()
for pin in (12, 18, 13, 19, 26, 16):
    try:
        pi.set_PWM_dutycycle(pin, 0)
        pi.write(pin, 0)
    except:
        pass
pi.stop()
print("Motors stopped.")
'''
# Write script to remote file then execute
sftp = ssh.open_sftp()
with sftp.file('/tmp/stop_motors.py', 'w') as f:
    f.write(script)
sftp.close()

stdin, stdout, stderr = ssh.exec_command("python3 /tmp/stop_motors.py")
print("STDOUT:", stdout.read().decode())
print("STDERR:", stderr.read().decode())
ssh.close()
