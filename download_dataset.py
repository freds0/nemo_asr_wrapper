import os
import gdown
import tarfile

data_dir = '.'

output_file = 'CORAA_DATASET_SAMPLE.tar.bz'
# Download the dataset. This will take a few moments...
print("Downloading CORAA Dataset")
if not os.path.exists(os.path.join(data_dir, output_file)):
    id = "132ylX-eH1qsyuNuHPwnF6jRCDsXQeEck"
    gdown.download(id=id, output=output_file, quiet=False)
    print(f"Dataset downloaded at: {}".format(output_file))
else:
    print("Tarfile already exists.")

my_tar = tarfile.open(os.path.join(data_dir, output_file))
my_tar.extractall('./data') # specify which folder to extract to
my_tar.close()

print("Finished conversion.\n******")
