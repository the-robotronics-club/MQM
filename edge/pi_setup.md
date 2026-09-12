# Raspberry Pi 3 Model A+ Setup Guide (MQM Edge)

This guide covers setting up a Raspberry Pi 3 Model A+ with the Raspberry Pi Camera Module 3 to run the MQM edge capture script.

## 1. Flash the OS

1. Download the [Raspberry Pi Imager](https://www.raspberrypi.com/software/).
2. Select **Raspberry Pi OS (Legacy, 64-bit) Lite** or **Raspberry Pi OS (64-bit) Lite**. A headless OS is highly recommended to save resources.
3. In the Imager settings (the gear icon):
   - Set a hostname (e.g., `mqm-cam-1`).
   - Enable SSH (use password authentication or public key).
   - Set your username and password.
   - Configure wireless LAN (Enter your campus Wi-Fi SSID and Password).
4. Write to a high-quality microSD card (at least 16GB, Class 10).

## 2. Hardware Assembly

1. **Connect the Camera:** Lift the black tab on the Pi 3 A+'s camera port, insert the cable (silver contacts facing away from the ethernet/USB ports), and push the tab down. Connect the other end to the Camera Module 3.
2. Insert the flashed microSD card.
3. Power on the Raspberry Pi 3 Model A+ via the `PWR IN` micro-USB port using a high-quality 5V 2.5A (or better) power supply.

## 3. SSH into the Pi

Find the IP address of your Pi (check your router, or try pinging `mqm-cam-1.local`).

```bash
ssh username@mqm-cam-1.local
```

## 4. Install Dependencies

The `libcamera` suite should already be installed on newer Raspberry Pi OS versions. You only need the `requests` library for the Python script.

```bash
sudo apt update
sudo apt install -y python3-requests
```

*Note: Camera Module 3 uses libcamera out of the box on Bullseye/Bookworm based Raspberry Pi OS. You do not need to enable legacy camera support in `raspi-config`.*

## 5. Deploy the Capture Script

Copy `edge/capture.py` from this repository to your Raspberry Pi.

```bash
# On your local machine (within the MQM project):
scp edge/capture.py username@mqm-cam-1.local:~/capture.py
```

## 6. Run the Script

On the Raspberry Pi, you can run the script manually to test it:

```bash
# Set your environment variables (replace with your server's IP and actual API key)
export SERVER_URL="http://192.168.1.100:8000"
export CAMERA_API_KEY="your-secret-key-here"
export CAMERA_ID="mess_main"

python3 capture.py
```

## 7. Run on Boot (Systemd Service)

To ensure the script runs automatically when the Pi boots up and restarts on failure, create a systemd service.

```bash
sudo nano /etc/systemd/system/mqm-capture.service
```

Add the following (replace `username`, `SERVER_URL`, and `CAMERA_API_KEY`):

```ini
[Unit]
Description=MQM Edge Camera Capture
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=your_username
Environment="SERVER_URL=http://192.168.1.100:8000"
Environment="CAMERA_API_KEY=your-secret-key-here"
Environment="CAMERA_ID=mess_main"
ExecStart=/usr/bin/python3 /home/your_username/capture.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Enable and start the service:

```bash
sudo systemctl enable mqm-capture.service
sudo systemctl start mqm-capture.service
```

You can check the logs with:

```bash
sudo journalctl -u mqm-capture.service -f
```
