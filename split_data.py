import os
import shutil

dirname = os.path.dirname(__file__)
data_directory = os.path.join(dirname, 'data')

data_size = 250

i = 0
current_directory = ''


os.mkdir('zip')
for f in os.listdir(data_directory):
    if i % data_size is 0:
        current_directory = 'data-' + str(i)
        os.mkdir('zip/' + current_directory)

    shutil.copy('data/' + f, 'zip/' + current_directory)

    i = i + 1