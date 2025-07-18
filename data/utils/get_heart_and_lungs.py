import os

print("This script will download all files required for benchmark problems."
      " You must have 'wget' installed for the downloads to work.")

folder_name = "data/heart_and_lungs"
if not os.path.isdir(folder_name):
    os.mkdir(folder_name)

fnames = ["BodyEIT.h5"]
for fid, fname in enumerate(fnames):
    print('Downloading file {fname} ({fid+1}/{len(fnames)}):')
    url = "https://zenodo.org/record/8121672/files/" + fname
    cmd = f"wget --directory-prefix {folder_name} {url}"
    os.system(cmd)