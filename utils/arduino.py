import serial
import serial.tools.list_ports
import time

def find_stm32_port():
    ports = serial.tools.list_ports.comports()
    for port in ports:
        if ("STM" in port.description or 
            "STLink" in port.description or
            "ttyACM" in port.device or
            "ttyUSB" in port.device):
            return port.device
    raise Exception("STM32 포트를 찾을 수 없습니다.")

def send_command():
    port = find_stm32_port()
    ser = serial.Serial(port, 9600)
    time.sleep(2)
    # ser.write(("ON" + "\n").encode("utf-8"))
    ser.write(b'1')
    print("✅ 간식 지급")
    ser.close()