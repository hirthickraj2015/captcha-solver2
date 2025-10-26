import os
import requests
import time

# --- Basic Setup ---
SHORTNAME = "dheenadh"
BASE_URL = "https://cs7ns1.scss.tcd.ie/"
FILE_LIST = "dheenadh-challenge-filenames"
OUTPUT_DIR = "dheenadh-dataset"

MAX_RETRIES = 3
RETRY_DELAY = 3      # seconds between retries of a single file
LOOP_DELAY = 10      # seconds before rechecking missing files

# --- Ensure output directory exists ---
if not os.path.isdir(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"Created folder: {OUTPUT_DIR}")
else:
    print(f"Using existing folder: {OUTPUT_DIR}")

# --- Load file list ---
try:
    with open(FILE_LIST, "r") as file:
        file_names = [line.strip() for line in file if line.strip()]
except FileNotFoundError:
    print(f"Cannot find file list: {FILE_LIST}")
    exit(1)

if not file_names:
    print("No files to process. Check the file list.")
    exit(0)

total_files = len(file_names)
print(f"Loaded {total_files} filenames to download.\n")

# --- Loop until all files are downloaded ---
while True:
    downloaded = [f for f in file_names if os.path.exists(os.path.join(OUTPUT_DIR, f))]
    missing = [f for f in file_names if f not in downloaded]

    print(f"{len(downloaded)}/{total_files} files downloaded. {len(missing)} remaining.")

    if not missing:
        print("All files successfully downloaded.")
        break

    for fname in missing:
        out_path = os.path.join(OUTPUT_DIR, fname)
        file_url = f"{BASE_URL}?shortname={SHORTNAME}&myfilename={fname}"

        success = False
        for attempt in range(1, MAX_RETRIES + 1):
            print(f"Downloading '{fname}' (attempt {attempt}/{MAX_RETRIES})...")

            try:
                res = requests.get(file_url, timeout=30)
                res.raise_for_status()

                with open(out_path, "wb") as save_file:
                    save_file.write(res.content)

                print(f"Downloaded: {fname}")
                success = True
                break

            except requests.exceptions.RequestException as err:
                print(f"Error downloading {fname}: {err}")
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY)
                else:
                    print(f"Failed to download {fname} after {MAX_RETRIES} attempts.")

        if not success:
            print(f"{fname} was not downloaded. Will retry in next loop.")

    # Wait a bit before checking missing files again
    time.sleep(LOOP_DELAY)
