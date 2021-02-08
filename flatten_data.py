import os
import shutil

output_directory = 'data-zips'
data_directory = 'data'

dirname = os.path.dirname(__file__)
data_directory_path = os.path.join(dirname, data_directory)
output_directory_path = os.path.join(dirname, output_directory)

os.mkdir(output_directory)
for dir in os.listdir(data_directory_path):
    current_directory = os.path.join(data_directory_path, dir)
    print("current directory: " + current_directory)
    for file in os.listdir(current_directory):
        print("current file: " + file)
        shutil.copy(str(current_directory) + '/' + file, output_directory)
