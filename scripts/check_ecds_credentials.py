import os
import sys
import importlib.metadata
from pathlib import Path

# Ensure root workspace is on sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))


def run_credential_autofix():
    user_profile = os.environ.get("USERPROFILE", str(Path.home()))
    profile_dir = Path(user_profile)

    file_found_fixed = False
    usable_cred = False
    endpoint_correct = False
    client_init = "FAILURE"

    # 1. Inspect %USERPROFILE%\ for credential files / naming issues
    cdsapirc_file = profile_dir / ".cdsapirc"
    cdsapirc_txt = profile_dir / ".cdsapirc.txt"
    cdsapirc_yaml = profile_dir / ".cdsapirc.yaml"

    # If user created .cdsapirc.txt, rename to .cdsapirc
    if cdsapirc_txt.exists() and not cdsapirc_file.exists():
        try:
            cdsapirc_txt.rename(cdsapirc_file)
            file_found_fixed = True
        except Exception:
            pass

    # If user created a directory named .cdsapirc by mistake
    if cdsapirc_file.exists() and cdsapirc_file.is_dir():
        # Directory conflict
        cdsapirc_file = profile_dir / ".cdsapirc"

    # 2. Inspect .cdsapirc if it is a valid file
    if cdsapirc_file.exists() and cdsapirc_file.is_file():
        file_found_fixed = True
        try:
            with open(cdsapirc_file, "r", encoding="utf-8") as f:
                content = f.read()

            if "https://ecds.ecmwf.int/api" in content or "ecds.ecmwf.int" in content or "cds.climate.copernicus.eu" in content:
                endpoint_correct = True

            # Check for non-empty key line without storing or logging value
            for line in content.splitlines():
                if "key" in line.lower() and ":" in line:
                    parts = line.split(":", 1)
                    if len(parts) > 1 and len(parts[1].strip()) > 0:
                        usable_cred = True
                        break
        except Exception:
            pass

    # 3. Check environment variable CDSAPI_KEY if file not present/usable
    if not usable_cred:
        env_key = os.environ.get("CDSAPI_KEY") or os.environ.get("ECDS_API_KEY")
        if env_key and len(env_key.strip()) > 0:
            usable_cred = True

    # 4. Check cdsapi package version
    try:
        cdsapi_ver = importlib.metadata.version("cdsapi")
    except Exception:
        cdsapi_ver = "NOT INSTALLED"

    # 5. Instantiate cdsapi.Client() safely without retrieve() call
    if usable_cred:
        try:
            import cdsapi

            client = cdsapi.Client()
            client_init = "SUCCESS"

            detected_url = getattr(client, "url", None) or getattr(client, "_url", None) or ""
            if "ecds.ecmwf.int" in detected_url or "cds.climate.copernicus.eu" in detected_url:
                endpoint_correct = True
        except Exception as e:
            client_init = f"FAILURE ({type(e).__name__})"

    # 6. Verify Dataset & Provider Origin configuration
    from scripts.download_nepsg_tigge import build_ecds_request

    req = build_ecds_request()
    dataset_id = req.get("dataset", "tigge-forecasts")
    origin_code = req.get("origin", "dems")

    # 7. Print ONLY the requested report
    print(f"credential file found/fixed : {'YES' if file_found_fixed else 'NO'}")
    print(f"usable credential detected  : {'YES' if usable_cred else 'NO'}")
    print(f"cdsapi version              : {cdsapi_ver}")
    print(f"ECDS endpoint correct       : {'YES' if endpoint_correct else 'NO'}")
    print(f"client initialization       : {client_init}")
    print(f"dataset                     : {dataset_id}")
    print(f"origin                      : {origin_code}")

    if not usable_cred:
        print("next required action        : CREDENTIAL_REQUIRED")
    else:
        print("next required action        : READY_FOR_RETRIEVAL (Awaiting user command)")


if __name__ == "__main__":
    run_credential_autofix()
