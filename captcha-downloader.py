import os
import requests
import time

# --- Basic Setup ---
# Just keeping these up here so I can tweak them easily later if needed.

SHORTNAME = "dheenadh"
BASE_URL = "https://cs7ns1.scss.tcd.ie/"
FILE_LIST = "dheenadh-challenge-filenames"
OUTPUT_DIR = "dheenadh-dataset"

# Retry settings — not sure if these need tuning, but this seems reasonable for now
MAX_RETRIES = 3
RETRY_DELAY = 3  # seconds to pause before retrying

# Create the dataset folder if it’s missing.
# (I always forget to do this, so putting it early.)
if not os.path.isdir(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"Created folder: {OUTPUT_DIR}")
else:
    # Just a small reassurance message
    print(f"Using existing folder: {OUTPUT_DIR}")

# --- Load the list of files to download ---
try:
    with open(FILE_LIST, "r") as file:
        file_names = [ln.strip() for ln in file if ln.strip()]
except FileNotFoundError:
    print(f"Can't find file list: {FILE_LIST}")
    file_names = []

# Just in case we got nothing
if not file_names:
    print("No files to process. (Maybe check the list file?)")
    exit(0)

# --- Download section ---
for fname in file_names:
    # Build output path
    out_path = os.path.join(OUTPUT_DIR, fname)

    # Skip existing files (no need to redownload)
    if os.path.exists(out_path):
        print(f"Skipping {fname} — already exists.")
        continue

    # Construct the full download URL
    file_url = f"{BASE_URL}?shortname={SHORTNAME}&myfilename={fname}"

    success = False  # just tracking manually
    for attempt_num in range(1, MAX_RETRIES + 1):
        print(f"Trying to download '{fname}' (attempt {attempt_num}/{MAX_RETRIES})...")

        try:
          
            res = requests.get(file_url, timeout=30)
            res.raise_for_status()

         
            with open(out_path, "wb") as save_file:
                save_file.write(res.content)

            print(f"Done: {out_path}")
            success = True
            break  

        except requests.exceptions.RequestException as err:
            print(f"Error downloading {fname}: {err}")
            if attempt_num < MAX_RETRIES:
                print(f"Will retry in {RETRY_DELAY} seconds...")
                time.sleep(RETRY_DELAY)
            else:
                print(f"Giving up on {fname} after {MAX_RETRIES} tries.")
        except Exception as generic_err:
            print(f"Unexpected error: {generic_err}")
            break

    if not success:
        print(f" File '{fname}' was not downloaded successfully.\n")
