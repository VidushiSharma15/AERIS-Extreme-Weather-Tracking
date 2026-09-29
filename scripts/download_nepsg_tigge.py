import os
import sys
import json
import hashlib
from pathlib import Path

# Ensure root workspace is on sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

RAW_NEPSG_DIR = Path("D:/SIH26078_AERIS/data/raw/nepsg")
TARGET_GRIB = RAW_NEPSG_DIR / "nepsg_amphan_2020.grib2"
MANIFEST_FILE = RAW_NEPSG_DIR / "manifest.json"
MAX_SAFETY_BYTES = 500 * 1024 * 1024  # 500 MB Hard Safety Limit


def calculate_sha256(file_path: Path) -> str:
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def build_ecds_request() -> dict:
    """
    Constructs the ECDS CDS-API request payload for the approved NEPS-G Amphan slice.
    """
    return {
        "dataset": "tigge-forecasts",
        "origin": "dems",
        "year": "2020",
        "month": "05",
        "day": "17",
        "time": "00:00",
        "level_type": "single_level",
        "variable": [
            "total_precipitation",
            "10_m_u_component_of_wind",
            "10_m_v_component_of_wind",
            "mean_sea_level_pressure",
        ],
        "forecast_type": ["control_forecast", "perturbed_forecast"],
        "leadtime_hour": ["0", "6", "12", "18", "24", "30", "36", "42", "48", "54", "60", "66", "72"],
        "area": [25.0, 80.0, 10.0, 95.0],  # North, West, South, East
        "data_format": "grib",
    }


def run_nepsg_download():
    print("=" * 80)
    print("AERIS NEPS-G / TIGGE DATA RETRIEVAL ENGINE (ECDS CDS-API MIGRATED)")
    print("=" * 80)

    # 1. Check local ECDS credentials setup
    home = Path.home()
    cdsapirc = home / ".cdsapirc"
    cds_env_key = os.environ.get("CDSAPI_KEY")

    if not cdsapirc.exists() and not cds_env_key:
        print("[AUTHENTICATION REQUIREMENT]")
        print("ECDS / CDS-API credentials file (~/.cdsapirc) or CDSAPI_KEY environment variable not found.")
        print("\nTo retrieve NEPS-G data via ECMWF Data Store (ECDS):")
        print("1. Register a free user account at: https://cds.climate.copernicus.eu/")
        print("2. Obtain your personal API key/token from your profile page.")
        print("3. Create file C:\\Users\\hp\\.cdsapirc with content:")
        print("   url: https://ecds.ecmwf.int/api")
        print("   key: <YOUR_CDS_PERSONAL_ACCESS_TOKEN>")
        print("\nAlternatively, set environment variables CDSAPI_KEY and CDSAPI_URL.")
        print("\n[STOPPING - AUTHENTICATION SETUP REQUIRED]")
        return False

    RAW_NEPSG_DIR.mkdir(parents=True, exist_ok=True)

    request_payload = build_ecds_request()

    print(f"Target Save Location : {TARGET_GRIB}")
    print(f"ECDS Dataset Identifier: {request_payload['dataset']}")
    print(f"Provider WMO Code    : {request_payload['origin']} (NCMRWF)")
    print(f"Event Case           : Cyclone Amphan (May 2020)")
    print(f"Initialization Cycle : {request_payload['date']} {request_payload['time']} UTC")
    print(f"Lead Steps (0-72h)   : {request_payload['lead_time_hour']}")
    print(f"Selected Members     : 5 members (Control + Perturbed 1..4)")
    print(f"Geographic Subgrid   : Lat [10.0 N - 25.0 N], Lon [80.0 E - 95.0 E]")
    print(f"Hard Safety Limit    : 500 MB\n")

    try:
        import cdsapi
        client = cdsapi.Client()

        print("-> Submitting retrieval request to ECMWF Data Store (ECDS)...")
        dataset_name = request_payload.pop("dataset")
        client.retrieve(dataset_name, request_payload, str(TARGET_GRIB))

        # Check file size safety
        file_bytes = TARGET_GRIB.stat().st_size
        file_mb = file_bytes / (1024 * 1024)
        print(f"\nDownload completed successfully: {file_mb:.2f} MB ({file_bytes} bytes)")

        if file_bytes > MAX_SAFETY_BYTES:
            print(f"ERROR: Downloaded file size ({file_mb:.2f} MB) exceeded 500 MB safety limit! Removing file.")
            TARGET_GRIB.unlink()
            return False

        # Calculate checksum & create manifest
        checksum = calculate_sha256(TARGET_GRIB)

        manifest = {
            "dataset_name": "nepsg_amphan_2020_development_slice",
            "data_source_type": "NWP_FORECAST",
            "nwp_system": "NCMRWF NEPS-G",
            "provider_wmo_code": "dems",
            "source_repository": "ECMWF Data Store (ECDS) TIGGE Archive",
            "source_url": "https://ecds.ecmwf.int/api",
            "dataset_id": "tigge-forecasts",
            "initialization_time": "2020-05-17T00:00:00Z",
            "forecast_lead_times_hours": [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72],
            "ensemble_members": {
                "control": [0],
                "perturbed": [1, 2, 3, 4],
                "total": 5
            },
            "variables": ["total_precipitation", "10m_u_component_of_wind", "10m_v_component_of_wind", "mean_sea_level_pressure"],
            "bounding_box": {
                "min_lat": 10.0,
                "max_lat": 25.0,
                "min_lon": 80.0,
                "max_lon": 95.0
            },
            "resolution_deg": 0.5,
            "file_format": "GRIB2",
            "file_size_bytes": file_bytes,
            "file_size_mb": round(file_mb, 2),
            "sha256_checksum": checksum,
            "local_path": str(TARGET_GRIB),
            "license": "CC BY-NC 4.0 / TIGGE Terms",
            "disclaimer": "Experimental NEPS-G development slice for GNN prototype validation; not operational forecast skill claim."
        }

        with open(MANIFEST_FILE, "w") as f:
            json.dump(manifest, f, indent=2)

        print(f"Dataset Manifest Saved: {MANIFEST_FILE}")
        print(f"SHA-256 Checksum       : {checksum}")
        print("=" * 80)
        return True

    except Exception as e:
        print(f"\nDownload Execution Error: {type(e).__name__} : {e}")
        return False


if __name__ == "__main__":
    run_nepsg_download()
