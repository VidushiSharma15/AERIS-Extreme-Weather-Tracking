import os
import zipfile
from pathlib import Path

def create_project_zip():
    project_dir = Path("D:/SIH26078_AERIS")
    zip_path = project_dir / "SIH26078_AERIS_updated_master.zip"
    
    exclude_dirs = {
        ".venv",
        "node_modules",
        ".next",
        "__pycache__",
        ".git",
        ".pytest_cache",
        ".idea",
        ".vscode",
    }
    
    count = 0
    total_bytes = 0
    
    print(f"Creating updated ZIP archive at: {zip_path} ...")
    
    with zipfile.ZipFile(str(zip_path), "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(str(project_dir)):
            # Filter out excluded directories in-place
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            
            for file in files:
                file_path = Path(root) / file
                if file_path == zip_path or file.endswith(".pyc") or file.endswith(".zip"):
                    continue
                
                arcname = file_path.relative_to(project_dir)
                zipf.write(str(file_path), str(arcname))
                count += 1
                total_bytes += file_path.stat().st_size
                
    zip_size_mb = zip_path.stat().st_size / (1024 * 1024)
    raw_size_mb = total_bytes / (1024 * 1024)
    
    print("=" * 80)
    print("AERIS MASTER PROJECT ZIP CREATED SUCCESSFULLY")
    print("=" * 80)
    print(f"ZIP Path       : {zip_path}")
    print(f"Files Packaged : {count} files")
    print(f"Uncompressed   : {raw_size_mb:.2f} MB")
    print(f"Compressed Size: {zip_size_mb:.2f} MB")
    print("=" * 80)

if __name__ == "__main__":
    create_project_zip()
