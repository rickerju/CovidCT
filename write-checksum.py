import csv
import os
import hashlib

dirname = os.path.dirname(__file__)
data_directory = os.path.join(dirname, 'data')

fileChecksums = []

for file in os.listdir(data_directory):
    with open(os.path.join(data_directory, file), "rb") as f:
        file_hash = hashlib.blake2b()
        chunk = f.read(8192)
        while chunk:
            file_hash.update(chunk)
            chunk = f.read(8192)

        fileChecksums.insert(0, (file, file_hash.hexdigest()))

with open(os.path.join(dirname, 'checksums.csv'), "w") as out:
    csv_out = csv.writer(out)
    csv_out.writerow(['file', 'checksum'])
    csv_out.writerows(fileChecksums)
