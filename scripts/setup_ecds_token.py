import os
import sys
import getpass
from pathlib import Path

def setup_token():
    user_profile = Path(os.environ.get("USERPROFILE", "C:\\Users\\hp"))
    rc_file = user_profile / ".cdsapirc"

    print("================================================================================")
    print("AERIS ECDS CREDENTIAL SECURE TERMINAL SETUP")
    print("================================================================================")
    
    token = getpass.getpass("ECDS Personal Access Token: ").strip()
    if not token:
        print("\n[ERROR] No token entered. Setup aborted.")
        sys.exit(1)

    try:
        with open(rc_file, "w", encoding="utf-8") as f:
            f.write("url: https://ecds.ecmwf.int/api\n")
            f.write(f"key: {token}\n")
        print("\n[SUCCESS] %USERPROFILE%\\.cdsapirc configured successfully.")
    except Exception as e:
        print(f"\n[ERROR] Failed to write credential file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    setup_token()
