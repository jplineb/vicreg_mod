import os
import threading
import subprocess
import requests

import threading
import subprocess

def download_file(url):
    command = f'wget -N -c -np --user jplineb --password=mS-q.wXZJvwxM4d {url}'
    subprocess.call(command, shell=True)

def main():
    base_url = 'https://physionet.org/files/mimic-cxr-jpg/2.0.0/files/'
    file_urls = ["p"+str(num) for num in range(10,20)]

    threads = []
    for url in file_urls:
        full_url = base_url + url
        thread = threading.Thread(target=download_file, args=(full_url,))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

if __name__ == "__main__":
    main()