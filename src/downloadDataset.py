import gdown
url = "https://drive.google.com/drive/folders/1w1_LYYDIGpTIlY5pD-VamDVMOMDAF9aQ?usp=drive_link"
gdown.download_folder(url, quiet=False, use_cookies=True)
