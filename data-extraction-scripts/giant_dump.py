import requests
import os
import re
import time

DUMP_URL="https://dumps.wikimedia.org/enwiki/latest/"
DOWNLOAD_DIR="wiki_downloads"
DOWNLOADABLE_EXTENSIONS=('.bz2','.gz','.zip')

def get_download_links(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching: {e}")
        return []

    links = re.findall(r'href="([^"]*)"', response.text)
    
    download_links = []
    for link in links:
        if link.endswith(DOWNLOADABLE_EXTENSIONS) and not link.startswith(('.', '?')):
            full_url = requests.utils.urljoin(url, link)
            download_links.append((link, full_url))
    return download_links

def download(url, local_filename):
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    except requests.exceptions.RequestException as e:
        print(f"Failed to download {local_filename}: {e}")

def main():
    if not os.path.exists(DOWNLOAD_DIR):
        os.makedirs(DOWNLOAD_DIR)
    files_to_download = download(DUMP_URL)
    for filename, url in files_to_download:
        local_path = os.path.join(DOWNLOAD_DIR, filename)
        if os.path.exists(local_path):
            continue
        download(url, local_path)
        time.sleep(3)

if __name__=="__main__":
    main()
