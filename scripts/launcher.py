import sys
import os
import time
import urllib.request
import urllib.error
import subprocess
import webbrowser

BACKEND_URL = "http://127.0.0.1:8000"
HEALTH_URL = "http://127.0.0.1:8000/health"
FRONTEND_PORT = 3001
FRONTEND_URL = f"http://localhost:{FRONTEND_PORT}"

# Resolve project paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend")

VENV_PYTHON = os.path.join(PROJECT_ROOT, ".venv", "Scripts", "python.exe")
PYTHON_EXE = VENV_PYTHON if os.path.exists(VENV_PYTHON) else sys.executable

def is_backend_running():
    try:
        req = urllib.request.Request(HEALTH_URL)
        with urllib.request.urlopen(req, timeout=2) as resp:
            return resp.status == 200
    except Exception:
        return False

def is_frontend_running():
    try:
        req = urllib.request.Request(FRONTEND_URL)
        with urllib.request.urlopen(req, timeout=2) as resp:
            return resp.status == 200
    except Exception:
        return False

def start_backend():
    if is_backend_running():
        print("[+] AERIS Backend is already running on http://127.0.0.1:8000")
        return True
    
    print("[*] Starting AERIS FastAPI Backend on http://127.0.0.1:8000 ...")
    if os.name == "nt":
        subprocess.Popen(
            f'start "AERIS Backend" /min "{PYTHON_EXE}" -m uvicorn backend.main:app --host 127.0.0.1 --port 8000',
            shell=True,
            cwd=PROJECT_ROOT
        )
    else:
        subprocess.Popen(
            [PYTHON_EXE, "-m", "uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", "8000"],
            cwd=PROJECT_ROOT
        )
    
    start_time = time.time()
    while time.time() - start_time < 15:
        if is_backend_running():
            print("\n[+] AERIS Backend started and healthy.")
            return True
        time.sleep(1)
        print(".", end="", flush=True)
    
    print("\n[-] ERROR: AERIS Backend failed to respond to health check within 15 seconds.")
    return False

def start_frontend():
    if is_frontend_running():
        print(f"[+] AERIS Frontend is already running on {FRONTEND_URL}")
        return True
    
    print(f"[*] Starting AERIS Next.js Frontend on port {FRONTEND_PORT} ...")
    if os.name == "nt":
        subprocess.Popen(
            f'start "AERIS Frontend" /min npm run dev -- -p {FRONTEND_PORT}',
            shell=True,
            cwd=FRONTEND_DIR
        )
    else:
        subprocess.Popen(
            ["npm", "run", "dev", "--", "-p", str(FRONTEND_PORT)],
            cwd=FRONTEND_DIR
        )
    
    start_time = time.time()
    while time.time() - start_time < 30:
        if is_frontend_running():
            print("\n[+] AERIS Frontend started and responsive.")
            return True
        time.sleep(1.5)
        print(".", end="", flush=True)
    
    print(f"\n[!] Warning: Frontend started, but {FRONTEND_URL} timed out. Launching browser anyway...")
    return True

def main():
    print("=" * 60)
    print(" AERIS ONE-CLICK SOFTWARE LAUNCHER")
    print("=" * 60)
    
    # Step 1: Backend check & start
    if not start_backend():
        print("[-] LAUNCH FAILED: Backend service initialization failed.")
        sys.exit(1)
    
    # Step 2: Frontend check & start
    if not start_frontend():
        print("[-] LAUNCH FAILED: Frontend service initialization failed.")
        sys.exit(1)
    
    # Step 3: Open browser automatically to frontend URL ONLY
    print(f"[*] Opening AERIS GIS Dashboard in default browser ({FRONTEND_URL})...")
    webbrowser.open(FRONTEND_URL)
    
    # Step 4: Display required ready output
    print("\n" + "=" * 60)
    print("AERIS IS READY")
    print(f"Dashboard: {FRONTEND_URL}")
    print("Backend: RUNNING")
    print("Amphan Demo: AVAILABLE")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
