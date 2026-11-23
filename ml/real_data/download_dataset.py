import urllib.request
import zipfile
import ssl
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config

UCI_URL = "https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip"

def download_and_extract() -> bool:
    """
    Attempts to download the UCI AI4I 2020 dataset and extract ai4i2020.csv into data/external/.
    Gracefully exits with clear instructions if network or URL fails.
    """
    config.EXTERNAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    target_csv = config.REAL_DATA_PATH

    if target_csv.exists() and target_csv.stat().st_size > 1000:
        print(f"[Dataset] AI4I 2020 dataset already exists at {target_csv}")
        return True

    zip_path = config.EXTERNAL_DATA_DIR / "ai4i2020.zip"
    print(f"[Dataset] Attempting download from {UCI_URL}...")

    try:
        ctx = ssl.create_default_context()
        req = urllib.request.Request(UCI_URL, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
        with urllib.request.urlopen(req, context=ctx, timeout=20) as resp, open(zip_path, "wb") as out_f:
            out_f.write(resp.read())
        print(f"[Dataset] Download completed. Extracting to {config.EXTERNAL_DATA_DIR}...")

        with zipfile.ZipFile(zip_path, "r") as z:
            # Look for ai4i2020.csv inside archive
            csv_candidates = [name for name in z.namelist() if name.endswith(".csv")]
            if csv_candidates:
                extracted_name = csv_candidates[0]
                z.extract(extracted_name, config.EXTERNAL_DATA_DIR)
                extracted_path = config.EXTERNAL_DATA_DIR / extracted_name
                if extracted_path != target_csv:
                    if target_csv.exists():
                        target_csv.unlink()
                    extracted_path.rename(target_csv)
                print(f"[Dataset] Successfully extracted {target_csv}")
                return True
            else:
                print("[Dataset] Warning: No .csv found inside downloaded zip.")
    except Exception as e:
        print(f"[Dataset] Download failed: {e}")
        print("=" * 70)
        print("MANUAL DOWNLOAD INSTRUCTIONS:")
        print("Please download the AI4I 2020 Predictive Maintenance CSV from the UCI repository")
        print("or Kaggle and save it directly as:")
        print(f"    {target_csv}")
        print("=" * 70)
        return False
    finally:
        if zip_path.exists():
            try:
                zip_path.unlink()
            except Exception:
                pass

    return target_csv.exists()

if __name__ == "__main__":
    download_and_extract()
