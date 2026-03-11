@echo off
cd /d "%~dp0"
echo [%date% %time%] Sync started >> sync_log.txt
python auto_sync.py >> sync_log.txt 2>&1
echo [%date% %time%] Sync finished >> sync_log.txt
