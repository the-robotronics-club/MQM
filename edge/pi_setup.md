# Raspberry Pi to the live Render dashboard

The Pi captures and uploads JPEGs. Inference runs on Render's free CPU using
the frozen ONNX models. The Pi needs no model files, PyTorch, or OpenCV.

- API: https://mqm-free-snapshot-api.onrender.com
- Dashboard: https://mqm-free-snapshot-dashboard.onrender.com
- Camera ID: `mess_main`

These steps assume Raspberry Pi OS with a supported CSI camera, such as Camera
Module 3. Use a current Raspberry Pi OS Lite image, enable SSH and Wi-Fi in
Raspberry Pi Imager, and attach the camera with power disconnected. Follow the
[official hardware instructions](https://www.raspberrypi.com/documentation/accessories/camera.html)
for your board/cable. A USB webcam needs a different capture command; this
script currently uses `rpicam-still` or older `libcamera-still`.

## 1. Install the lightweight uploader

Run these commands **on the Pi**, over SSH or in its terminal:

```bash
sudo apt update
sudo apt install -y python3-requests rpicam-apps curl ca-certificates
mkdir -p "$HOME/mqm"
curl --fail --location \
  https://raw.githubusercontent.com/arywk40-hue/MQM/main/edge/capture.py \
  --output "$HOME/mqm/capture.py"
```

Check that the camera is detected and can capture a frame:

```bash
rpicam-hello --list-cameras
rpicam-still -n --width 1280 --height 720 --timeout 1500 -o "$HOME/mqm/test.jpg"
```

The modern command names and camera-list option are documented in
[Raspberry Pi's camera software guide](https://www.raspberrypi.com/documentation/computers/camera_software.html).
If no camera is listed, fix the cable/camera detection before proceeding.

## 2. Configure the cloud connection

Open the API service's
[Render environment panel](https://dashboard.render.com/web/srv-dbctjqdg1s2s73egder0/env).
Copy **only `CAMERA_API_KEY`**. Paste it into the hidden terminal prompt below;
the actual key must not be committed or included in screenshots.

```bash
export SERVER_URL=https://mqm-free-snapshot-api.onrender.com
export CAMERA_ID=mess_main
export CAPTURE_INTERVAL_SECONDS=10
export UPLOAD_TIMEOUT_SECONDS=180
read -r -s -p 'Paste CAMERA_API_KEY from Render: ' CAMERA_API_KEY
printf '\n'
export CAMERA_API_KEY
```

Use the API URL above for uploads; the dashboard is a separate read-only app.
The `read` command assumes Bash, the usual Raspberry Pi OS terminal shell.

## 3. Test once, then start continuous capture

First send the JPEG captured above:

```bash
python3 "$HOME/mqm/capture.py" --image "$HOME/mqm/test.jpg" --once
```

Expect `Stored: people=... at ...`. Check that the dashboard displays that
timestamp and count. This tests your Pi's actual network/upload path.

Then test one direct camera capture and start the loop:

```bash
python3 "$HOME/mqm/capture.py" --once
python3 "$HOME/mqm/capture.py"
```

The loop captures 1280x720 JPEGs, sends them via HTTPS with `X-Camera-Key`, and
waits for the server's stored-reading acknowledgement. It aims for a 10-second
interval; slow inference/network requests extend it, with no overlapping
uploads. Use Ctrl+C to stop. Change `IMAGE_WIDTH` and `IMAGE_HEIGHT` if needed;
keep an upright, stable view of the cafeteria for the demonstration.

## 4. Start automatically after reboot (optional)

After the manual check works, save the current configuration in a private file:

```bash
python3 - <<'PY'
import os
from pathlib import Path
keys = ('SERVER_URL', 'CAMERA_ID', 'CAMERA_API_KEY',
        'CAPTURE_INTERVAL_SECONDS', 'UPLOAD_TIMEOUT_SECONDS')
config = Path.home() / 'mqm' / 'camera.env'
config.touch(mode=0o600, exist_ok=True)
config.chmod(0o600)
config.write_text(''.join(f'{key}={os.environ[key]}\n' for key in keys))
PY
```

Create the service using your current username and home directory:

```bash
sudo tee /etc/systemd/system/mqm-capture.service >/dev/null <<EOF
[Unit]
Description=MQM camera snapshots to Render
Wants=network-online.target
After=network-online.target

[Service]
Type=simple
User=$(id -un)
WorkingDirectory=$HOME/mqm
EnvironmentFile=$HOME/mqm/camera.env
ExecStart=/usr/bin/python3 $HOME/mqm/capture.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload
sudo systemctl enable --now mqm-capture.service
sudo journalctl -u mqm-capture.service -f
```

Stop any manually running uploader before enabling the service. To stop it:

```bash
sudo systemctl stop mqm-capture.service
```

## What is verified and what remains

On October 22, cloud inference returned zero for a synthetic blank image and
four people for Ultralytics' public bus sample. New counts/timestamps matched
the API's current status, stored history, and the actual hosted dashboard.
The public sample was grouped as four queued, zero sitting, zero other standing;
this is transport/inference proof, not a labeled cafeteria accuracy evaluation.
Nearby groups can be mistaken for queues by the distance rule.

Physical Pi capture and accuracy in the mounted cafeteria view remain untested.
The dashboard refreshes every eight seconds. With no successful uploads for
120 seconds, the camera becomes offline. Free Render instances may sleep and
have slow cold starts; SQLite history resets when the instance filesystem
resets. Warm both services before the demonstration. The free workspace's
included hours are shared with its other free services.
