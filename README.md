## 1. Windows 환경
### ✅ **Python**
- [download](https://www.python.org/downloads/)
- version: Python3.12.3

### ☑️ **pip**
```bash
# wsl
sudo apt install python3-pip
```
<!-- - version: pip 24.0 from /usr/lib/python3/dist-packages/pip (python 3.12)  -->

### ☑️ **python3.12-venv**
```bash
# wsl
sudo apt install python3.12-venv
```

### ☑️ **ultralytics**
```bash
# wsl
python3 -m venv myenv
source myenv/bin/activate
pip install ultralytics
```
- version: Ultralytics 8.3.94 🚀 Python-3.12.3

### ✅ **usbipd-win**
```bash
# powershell
winget install --interactive --exact dorssel.usbipd-win
```
```bash
# wsl
sudo apt install linux-tools-generic
```
```bash
# Windows Powershell (Administrator)
usbipd list

usbipd bind --busid 1-2

usbipd attach --busid 1-2 --wsl
```
```plainText
Connected:
BUSID  VID:PID    DEVICE                                                        STATE
1-2    0c45:6340  USB Camera, USB Microphone                                    Attached
1-4    0489:e0f5  MediaTek Bluetooth Adapter                                    Not shared
4-4    0bda:8153  Realtek USB GbE Family Controller #2                          Not shared
5-2    05ac:024f  USB 입력 장치                                                 Not shared
5-3    046d:c548  Logitech USB Input Device, USB 입력 장치                      Not shared
6-1    3277:0060  USB2.0 FHD UVC WebCam, USB2.0 IR UVC WebCam, Camera DFU D...  Not shared
```
```bash
# wsl
sudo apt install -y usbutils v4l-utils
lsusb
```
```plainText
Bus 001 Device 001: ID 1d6b:0002 Linux Foundation 2.0 root hub
Bus 001 Device 002: ID 0c45:6340 Microdia Camera
Bus 002 Device 001: ID 1d6b:0003 Linux Foundation 3.0 root hub
```
## 2. Linux 환경

