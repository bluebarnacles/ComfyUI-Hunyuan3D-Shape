import subprocess
import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

print("[Hunyuan3D-Shape] Installing pip dependencies...")
requirements_path = os.path.join(script_dir, "requirements.txt")
try:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_path])
except subprocess.CalledProcessError as e:
    print(f"[Hunyuan3D-Shape] WARNING: Some pip dependencies failed to install: {e}")
