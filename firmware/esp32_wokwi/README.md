# ESP32 Virtual Firmware Setup (Wokwi Simulation)

This directory contains production-ready ESP32 firmware and wiring diagrams for the Smart Factory Digital Twin.

## Hardware Wiring
- **ESP32 NodeMCU DevKit V4**
- **DHT22 (Temperature)**: Data pin -> GPIO 15, VCC -> 3.3V, GND -> GND
- **MPU6050 (Vibration / 3-axis Accelerometer)**: SDA -> GPIO 21, SCL -> GPIO 22, VCC -> 3.3V, GND -> GND
- **Potentiometer (Current/Power Sensor Proxy)**: SIG -> GPIO 34 (ADC1), VCC -> 3.3V, GND -> GND

## How to Run on Wokwi (Web Simulator)
1. Open [https://wokwi.com](https://wokwi.com) and create a new **ESP32** project.
2. Replace `sketch.ino` with the contents of `firmware/esp32_wokwi/sketch.ino`.
3. Replace `diagram.json` with `firmware/esp32_wokwi/diagram.json`.
4. Add the libraries listed in `libraries.txt` in the Wokwi Library Manager tab.
5. Hit the **Start Simulation** button.

## Important Note on Localhost Connectivity
> [!NOTE]
> Because Wokwi runs in your web browser cloud sandbox, **it cannot directly reach `http://localhost:5000` or `127.0.0.1`** on your computer.

To bridge data into your local Digital Twin:
1. **Option A (Serial Bridge - Recommended for zero setup & real hardware):**
   Copy the JSON lines from Wokwi's serial monitor or run:
   ```bash
   python scripts/serial_bridge.py --mock
   ```
   Or connect your physical ESP32 to USB and run:
   ```bash
   python scripts/serial_bridge.py --port COM3 --baud 115200
   ```
2. **Option B (Public Tunnel):**
   Start an ngrok or cloudflare tunnel pointing to your Flask API:
   ```bash
   ngrok http 5000
   ```
   Update `API_URL` at the top of `sketch.ino` with your public ngrok URL (`https://<id>.ngrok-free.app/sensor-data`).
