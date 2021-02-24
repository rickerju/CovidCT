import csv
import os
import hashlib
import pandas

dirname = os.path.dirname(__file__)
data_directory = os.path.join(dirname, 'data')

erroredFiles = []
fileChecksums = []

for file in os.listdir(data_directory):
    with open(os.path.join(data_directory, file), "rb") as f:
        file_hash = hashlib.blake2b()
        chunk = f.read(8192)
        while chunk:
            file_hash.update(chunk)
            chunk = f.read(8192)

        fileChecksums.insert(0, (file, file_hash.hexdigest()))

current_checksums = pandas.DataFrame(fileChecksums, columns=["file", "checksum"])
valid_checksums = pandas.read_csv(os.path.join(dirname, 'checksums.csv'))
for valid_row in valid_checksums.itertuples(index=False):
    current_row = current_checksums.loc[(current_checksums['file'] == valid_row[0])
                                        & (current_checksums['checksum'] == valid_row[1])]

    if current_row.shape[0] == 0:
        print(f"row not found for {valid_row[0]}")
        erroredFiles.insert(0, valid_row)

    elif current_row.shape[0] > 1:
        print(f"{current_row.size} rows found for {valid_row[0]}")
        erroredFiles.insert(0, valid_row)

with open(os.path.join(dirname, 'checksum-errors.csv'), "w") as out:
    csv_out = csv.writer(out)
    csv_out.writerow(['file', 'checksum'])
    csv_out.writerows(fileChecksums)