"""Serve the camera-only page over local HTTPS with a short-lived test CA."""

from __future__ import annotations

import argparse
import ipaddress
import os
import secrets
import subprocess
import sys
import tempfile
import threading
from datetime import UTC, datetime, timedelta
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID


def certificates(root: Path, address: str) -> tuple[Path, Path]:
    now = datetime.now(UTC)
    ca_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "MQM local camera test")])
    ca = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=5))
        .not_valid_after(now + timedelta(days=7))
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .add_extension(
            x509.KeyUsage(False, False, False, False, False, True, True, False, False),
            critical=True,
        )
        .sign(ca_key, hashes.SHA256())
    )
    public = root / "public"
    public.mkdir()
    (public / "mqm-camera-ca.cer").write_bytes(ca.public_bytes(serialization.Encoding.DER))
    (root / "ca.pem").write_bytes(ca.public_bytes(serialization.Encoding.PEM))
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    cert = (
        x509.CertificateBuilder()
        .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, address)]))
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=5))
        .not_valid_after(now + timedelta(days=7))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(
            x509.SubjectAlternativeName([x509.IPAddress(ipaddress.ip_address(address))]),
            critical=False,
        )
        .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
        .sign(ca_key, hashes.SHA256())
    )
    cert_path, key_path = root / "server.pem", root / "server-key.pem"
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    key_path.chmod(0o600)
    return cert_path, key_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--address", required=True, help="This Mac's private Wi-Fi IPv4 address")
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Also serve the full dashboard over HTTPS on port 8503",
    )
    args = parser.parse_args()
    address = ipaddress.IPv4Address(args.address)
    if not address.is_private or address.is_loopback or address.is_unspecified:
        parser.error("Use the Mac's private Wi-Fi address")
    repo = Path(__file__).resolve().parent.parent
    root = Path(tempfile.mkdtemp(prefix="mqm-phone-"))
    cert, key = certificates(root, str(address))
    code = secrets.token_urlsafe(9)
    server = ThreadingHTTPServer(
        (str(address), 8765), partial(SimpleHTTPRequestHandler, directory=str(root / "public"))
    )
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"Certificate: http://{address}:8765/mqm-camera-ca.cer", flush=True)
    print(f"Phone camera: https://{address}:8502", flush=True)
    print(f"Access code: {code}", flush=True)
    print(f"Certificate files: {root}", flush=True)
    print(
        "Certificate expires in 7 days. Stop this launcher with Ctrl-C after testing.", flush=True
    )
    env = {**os.environ, "PHONE_CAMERA_ACCESS_CODE": code, "PYTHONPATH": str(repo)}
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(repo / "dashboard/phone.py"),
            "--server.address",
            str(address),
            "--server.port",
            "8502",
            "--server.headless",
            "true",
            "--server.sslCertFile",
            str(cert),
            "--server.sslKeyFile",
            str(key),
        ],
        cwd=repo,
        env=env,
    )
    dashboard_process = None
    if args.dashboard:
        dashboard_process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                str(repo / "dashboard/app.py"),
                "--server.address",
                str(address),
                "--server.port",
                "8503",
                "--server.headless",
                "true",
                "--server.sslCertFile",
                str(cert),
                "--server.sslKeyFile",
                str(key),
            ],
            cwd=repo,
            env=env,
        )
        print(f"Full dashboard: https://{address}:8503", flush=True)
    try:
        process.wait()
    except KeyboardInterrupt:
        pass
    finally:
        process.terminate()
        process.wait(timeout=15)
        if dashboard_process is not None:
            dashboard_process.terminate()
            dashboard_process.wait(timeout=15)
        server.shutdown()


if __name__ == "__main__":
    main()
