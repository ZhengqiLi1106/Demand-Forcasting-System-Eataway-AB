"""
Eataway Auto-Sync Script
========================
Workflow:
  1. Pull latest data from company MySQL database (local)
  2. Run feature engineering + model training locally
     (avoids server memory limits — training is CPU/RAM intensive)
  3. Upload historical data to Railway (for historical queries)
  4. Upload prediction results to Railway (for the prediction UI)

Usage:
  python auto_sync.py

Scheduled runs: configure Windows Task Scheduler to run this script weekly.
"""

import sys
import time
import io
import os
import zipfile
import subprocess
import requests
import pandas as pd
import pymysql
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent
load_dotenv(dotenv_path=BASE_DIR / ".env")

DB_CONFIG = {
    "host":     os.environ["DB_HOST"],
    "port":     int(os.environ.get("DB_PORT", 3306)),
    "user":     os.environ["DB_USER"],
    "password": os.environ["DB_PASSWORD"],
    "database": os.environ["DB_NAME"],
    "charset":  "utf8mb4",
}

SITE_URL     = os.environ.get("SITE_URL", "https://web-production-858ef.up.railway.app")
APP_PASSWORD = os.environ.get("APP_PASSWORD", "eataway")
DAYS_HISTORY = 730

# ============================================================
# Script logic (no need to modify below)
# ============================================================

TABLE          = "ordrar_och_returer_looker_1y"
DATA_FILE      = BASE_DIR / "1year.csv"
OUTPUT_DIR     = BASE_DIR / "output_v7"
PRED_DIR       = BASE_DIR / "predictions"
FEATURE_SCRIPT = BASE_DIR / "feature.py"
TRAIN_SCRIPT   = BASE_DIR / "eataway_train_v7.py"


def fetch_from_db() -> pd.DataFrame:
    print("▶ Connecting to database...")
    conn = pymysql.connect(**DB_CONFIG)
    try:
        query = f"""
            SELECT datum, namn, ort, typ, sort, antal_ordrar, antal_returer
            FROM {TABLE}
            WHERE datum >= DATE_SUB(CURDATE(), INTERVAL {DAYS_HISTORY} DAY)
            ORDER BY datum
        """
        df = pd.read_sql(query, conn)
        print(f"  ✓ Fetched {len(df):,} rows, date range: {df['datum'].min()} ~ {df['datum'].max()}")
        return df
    finally:
        conn.close()


def run_local_training() -> bool:
    """Run feature engineering and model training locally."""
    python = sys.executable
    steps = [
        ("Feature Engineering", [python, str(FEATURE_SCRIPT)]),
        ("Model Training",      [python, str(TRAIN_SCRIPT)]),
    ]
    for name, cmd in steps:
        print(f"▶ Running locally: {name}...")
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
            cwd=str(BASE_DIR), env=env,
        )
        for line in proc.stdout:
            print(f"  {line.rstrip()}")
        proc.wait()
        if proc.returncode != 0:
            print(f"  ✗ {name} failed (code={proc.returncode})")
            return False
        print(f"  ✓ {name} complete")
    return True


def wake_server() -> bool:
    print("▶ Connecting to server...")
    for attempt in range(1, 7):
        try:
            r = requests.get(SITE_URL, timeout=30)
            if r.status_code == 200:
                print("  ✓ Server is ready")
                return True
        except requests.exceptions.RequestException:
            pass
        print(f"  Waiting... ({attempt}/6)")
        time.sleep(15)
    print("  ✗ Server not responding")
    return False


def upload_history(df: pd.DataFrame) -> bool:
    """Upload historical CSV for the history query page."""
    print("▶ Uploading historical data...")
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    r = requests.post(
        f"{SITE_URL}/api/upload",
        files={"file": ("1year.csv", csv_bytes, "text/csv")},
        params={"pw": APP_PASSWORD},
        timeout=300,
    )
    result = r.json()
    if result.get("ok"):
        print(f"  ✓ {result['msg']}")
        return True
    print(f"  ✗ Upload failed: {result.get('msg')}")
    return False


def upload_results() -> bool:
    """Zip output_v7/ and predictions/ and upload to Railway."""
    print("▶ Uploading prediction results to server...")
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in OUTPUT_DIR.glob("*.csv"):
            zf.write(f, f"output_v7/{f.name}")
        if PRED_DIR.exists():
            for f in PRED_DIR.glob("*"):
                if f.is_file():
                    zf.write(f, f"predictions/{f.name}")

    zip_bytes = zip_buf.getvalue()
    r = requests.post(
        f"{SITE_URL}/api/upload-results",
        files={"file": ("results.zip", zip_bytes, "application/zip")},
        params={"pw": APP_PASSWORD},
        timeout=120,
    )
    result = r.json()
    if result.get("ok"):
        print(f"  ✓ {result['msg']}")
        return True
    print(f"  ✗ Upload failed: {result.get('msg')}")
    return False


def main():
    print("=" * 55)
    print("  Eataway Auto-Sync  (local training mode)")
    print(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Target: {SITE_URL}")
    print("=" * 55)

    # 1. Pull data from database
    try:
        df = fetch_from_db()
    except Exception as e:
        print(f"✗ Database connection failed: {e}")
        sys.exit(1)

    # 2. Save locally (training scripts read from this file)
    print("▶ Saving local data file...")
    df.to_csv(DATA_FILE, index=False)
    print(f"  ✓ Saved to {DATA_FILE}")

    # 3. Train locally (no server memory limits)
    if not run_local_training():
        print("✗ Local training failed — check errors above")
        sys.exit(1)

    # 4. Wake server
    if not wake_server():
        sys.exit(1)

    # 5. Upload historical data
    if not upload_history(df):
        sys.exit(1)

    # 6. Upload prediction results
    if not upload_results():
        sys.exit(1)

    print(f"\n✓ All done! Site updated with latest predictions.")
    print(f"  Visit: {SITE_URL}")


if __name__ == "__main__":
    main()
