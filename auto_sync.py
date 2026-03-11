"""
Eataway Auto-Sync Script
========================
Workflow:
  1. Pull latest data from company MySQL database
  2. Upload to the web server
  3. Trigger server-side model retraining

Usage:
  python auto_sync.py

Scheduled runs: configure Windows Task Scheduler to run this script weekly.
"""

import sys
import time
import io
import os
import requests
import pandas as pd
import pymysql
from pathlib import Path
from dotenv import load_dotenv

# Load credentials from .env (passwords are not hardcoded here)
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

SITE_URL     = os.environ.get("SITE_URL", "https://eataway.up.railway.app")
APP_PASSWORD = os.environ.get("APP_PASSWORD", "eataway")
DAYS_HISTORY = 400

# ============================================================
# Script logic (no need to modify below)
# ============================================================

TABLE = "ordrar_och_returer_looker_1y"


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


def wake_server() -> bool:
    """Ping the server to ensure it is awake and ready."""
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


def upload_data(df: pd.DataFrame) -> bool:
    print("▶ Uploading data to server...")
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    r = requests.post(
        f"{SITE_URL}/api/upload",
        files={"file": ("1year.csv", csv_bytes, "text/csv")},
        params={"pw": APP_PASSWORD},
        timeout=180,
    )
    result = r.json()
    if result.get("ok"):
        print(f"  ✓ {result['msg']}")
        return True
    print(f"  ✗ Upload failed: {result.get('msg')}")
    return False


def trigger_training() -> bool:
    print("▶ Triggering model training...")
    r = requests.post(
        f"{SITE_URL}/api/run",
        json={"pw": APP_PASSWORD},
        timeout=30,
    )
    result = r.json()
    if result.get("ok"):
        print("  ✓ Training started")
        return True
    print(f"  ✗ Failed to start: {result.get('msg')}")
    return False


def wait_for_training() -> bool:
    print("▶ Waiting for training to complete (usually 3-10 minutes)...")
    start = time.time()
    while True:
        elapsed = int(time.time() - start)
        time.sleep(15)
        try:
            r = requests.get(f"{SITE_URL}/api/status", timeout=30)
            s = r.json()
            if s.get("done"):
                if s.get("error"):
                    print(f"  ✗ Training failed ({elapsed}s)")
                    for line in s.get("log", [])[-10:]:
                        print(f"    {line}")
                    return False
                print(f"  ✓ Training complete ({elapsed}s)")
                return True
            if elapsed % 30 == 0:
                print(f"  ... Training in progress {elapsed}s")
        except requests.exceptions.RequestException as e:
            print(f"  ! Status check failed: {e}, retrying...")
        if elapsed > 900:
            print("  ✗ Training timed out (15 min), please check the site manually")
            return False


def main():
    print("=" * 55)
    print("  Eataway Auto-Sync")
    print(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Target: {SITE_URL}")
    print("=" * 55)

    # 1. Pull data from database
    try:
        df = fetch_from_db()
    except Exception as e:
        print(f"✗ Database connection failed: {e}")
        sys.exit(1)

    # 2. Wake server
    if not wake_server():
        sys.exit(1)

    # 3. Upload data
    if not upload_data(df):
        sys.exit(1)

    # 4. Trigger training
    if not trigger_training():
        sys.exit(1)

    # 5. Wait for completion
    if not wait_for_training():
        sys.exit(1)

    print(f"\n✓ All done! Site updated with latest predictions.")
    print(f"  Visit: {SITE_URL}")


if __name__ == "__main__":
    main()
