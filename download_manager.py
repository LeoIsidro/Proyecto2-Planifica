import os
import sys
import json
import gdown
import fnmatch
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

FOLDER_ID = "1bAyIR3KvCbkhinDRHX2xeWgT3F_VSLAq"
MANIFEST_FILE = "download_manifest.json"
TARGET_DIR = "/home/leoisidro/CICLOS/X/PLANIFICA/proyecto_2"
MAX_WORKERS = 4  # Adjust based on bandwidth and performance
MAX_RETRIES = 3

PATTERNS = [
    "HD7/*_Study.zip",
    "HD7/*_Bedroom.zip",
    "HD7/*_Living_room.zip"
]

manifest_lock = threading.Lock()

def save_manifest(manifest):
    with manifest_lock:
        with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

def load_manifest():
    if os.path.exists(MANIFEST_FILE):
        with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def fetch_and_build_manifest():
    print("Fetching file list from Google Drive folder...")
    files = gdown.download_folder(id=FOLDER_ID, skip_download=True)
    print(f"Total files in folder: {len(files)}")
    
    manifest = []
    for f in files:
        path = f.path
        # Check if it matches any pattern
        matches_any = False
        for p in PATTERNS:
            if fnmatch.fnmatch(path, p) or fnmatch.fnmatch(path.lower(), p.lower()):
                matches_any = True
                break
        
        if matches_any:
            # Construct local output path
            local_path = os.path.join(TARGET_DIR, path)
            manifest.append({
                "id": f.id,
                "path": path,
                "local_path": local_path,
                "status": "pending",
                "retries": 0
            })
            
    print(f"Filtered manifest size: {len(manifest)} files matching patterns.")
    save_manifest(manifest)
    return manifest

def download_file(item):
    file_id = item["id"]
    local_path = item["local_path"]
    path = item["path"]
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    
    # Download
    print(f"[START] Downloading {path} ...")
    try:
        # gdown will overwrite or resume. We use quiet=True to avoid stdout clutter
        out = gdown.download(id=file_id, output=local_path, quiet=True)
        if out and os.path.exists(local_path):
            item["status"] = "completed"
            print(f"[SUCCESS] Finished {path}")
            return True
        else:
            raise Exception("gdown returned None or file not found")
    except Exception as e:
        item["retries"] += 1
        print(f"[ERROR] Failed to download {path} (Attempt {item['retries']}/{MAX_RETRIES}): {e}")
        if item["retries"] >= MAX_RETRIES:
            item["status"] = "failed"
        return False

def main():
    manifest = load_manifest()
    if not manifest:
        manifest = fetch_and_build_manifest()
        
    pending_items = [item for item in manifest if item["status"] == "pending"]
    completed_count = sum(1 for item in manifest if item["status"] == "completed")
    failed_count = sum(1 for item in manifest if item["status"] == "failed")
    
    total = len(manifest)
    print(f"\nStatus: {completed_count}/{total} completed, {failed_count}/{total} failed, {len(pending_items)} pending.")
    
    if not pending_items:
        print("No pending downloads. Everything is up to date!")
        return
        
    print(f"Starting parallel download with {MAX_WORKERS} workers...")
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(download_file, item): item for item in pending_items}
        
        for future in as_completed(futures):
            item = futures[future]
            # Save manifest after each download attempt to ensure progress is saved
            save_manifest(manifest)
            
            # Print current progress summary
            completed = sum(1 for x in manifest if x["status"] == "completed")
            failed = sum(1 for x in manifest if x["status"] == "failed")
            pending = sum(1 for x in manifest if x["status"] == "pending")
            print(f"[PROGRESS] Completed: {completed}/{total} | Failed: {failed}/{total} | Pending: {pending}/{total}")

if __name__ == "__main__":
    main()
