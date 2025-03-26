## Jetson Xavier

### ✅ **Python**
```bash
sudo apt install python3
```
🚀 version: Python 3.8.10

### ✅ **pip**
```bash
sudo apt install python3-pip
```
🚀 version: pip 20.0.2

### ☑️ **pytorch**
[https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048)

☝️ 위 링크에서 JetPack 5 Pytorch v2.0.0 wheel 다운로드
( jetpack v5.1.5에 맞는 최신 안정화 버전 ) 
```bash
# sudo apt show nvidia-jetpack 명령어로 jetpack 버전 확인 -> 5.1.5-b11
pip install ~/Download/torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl
```

### ☑️ **pyserial**
```bash
pip install pyserial
```
```bash
# ⚠️ Arduino Uno /dev/ttyACM0 Permission denied 해결
sudo setfacl -m u:nvidia:rw /dev/ttyACM0
```