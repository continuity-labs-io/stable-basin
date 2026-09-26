We need a Python script to programmatically download and extract a massive behavioral tracking dataset from Zenodo. 

Please create a script named `tools/download_killifish_aging.py` that does the following:

### 1. Query the Zenodo API
Query the specific Zenodo record endpoint: `https://zenodo.org/api/records/17238217` using the `requests` library. 
Parse the returned JSON object to access the `files` array.

### 2. Locate and Download the Data Archive
Iterate through the `files` array and find the file named exactly `data.zip`. 
* Extract its `links.self` download URL.
* Implement a robust download function using the `requests` library with `stream=True`. 
* Because this is an 11.8 GB file, it is critical to include a progress bar using `tqdm` that displays the download speed and percentage complete based on the `Content-Length` header.
* Save the file to `data/killifish/data.zip`.
* *Important:* Check if the file already exists locally and matches the expected size before downloading to avoid re-downloading 11.8 GB unnecessarily. 

### 3. Automatic Extraction
Once the download is complete (or if the file already exists), use the built-in `zipfile` module to automatically extract the contents of `data.zip` directly into the `data/killifish/` directory.

Please ensure the script creates any necessary parent directories (e.g., `os.makedirs('data/killifish', exist_ok=True)`) before attempting to save the file.
