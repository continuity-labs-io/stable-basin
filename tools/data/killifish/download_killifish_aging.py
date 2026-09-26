import os
import requests
import zipfile
from tqdm import tqdm

def download_file(url, dest_path, expected_size=None):
    if os.path.exists(dest_path):
        if expected_size:
            actual_size = os.path.getsize(dest_path)
            if actual_size == expected_size:
                print(f"File already exists and matches expected size ({expected_size} bytes). Skipping download.")
                return True
            else:
                print(f"File exists but size ({actual_size} bytes) does not match expected ({expected_size} bytes). Re-downloading.")
        else:
            print(f"File already exists. Skipping download.")
            return True

    print(f"Downloading from {url}")
    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get('content-length', 0))
    if expected_size and total_size and expected_size != total_size:
        print(f"Warning: Expected size {expected_size} but server reported {total_size}")
    
    # Use expected_size if total_size is 0 (some servers might not send content-length)
    if total_size == 0 and expected_size:
        total_size = expected_size

    block_size = 1024 * 1024 # 1 Megabyte
    with open(dest_path, 'wb') as f:
        with tqdm(total=total_size, unit='iB', unit_scale=True, desc="Downloading data.zip") as pbar:
            for data in response.iter_content(block_size):
                f.write(data)
                pbar.update(len(data))
    return True

def extract_zip(zip_path, extract_dir):
    print(f"Extracting {zip_path} to {extract_dir}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print("Extraction complete.")

def main():
    record_id = "17238217"
    api_url = f"https://zenodo.org/api/records/{record_id}"
    
    output_dir = "data/killifish"
    zip_dest = os.path.join(output_dir, "data.zip")
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Querying Zenodo API: {api_url}")
    response = requests.get(api_url)
    response.raise_for_status()
    
    data = response.json()
    
    files = data.get("files", [])
    
    target_file = None
    for file_info in files:
        if file_info.get("key") == "data.zip":
            target_file = file_info
            break
            
    if not target_file:
        print("Error: Could not find 'data.zip' in the Zenodo record.")
        return
        
    download_url = target_file.get("links", {}).get("self")
    if not download_url:
        print("Error: Could not find download link for 'data.zip'.")
        return
        
    expected_size = target_file.get("size")
    
    print(f"Found data.zip: size={expected_size} bytes, URL={download_url}")
    
    success = download_file(download_url, zip_dest, expected_size)
    
    if success:
        extract_zip(zip_dest, output_dir)

if __name__ == "__main__":
    main()
