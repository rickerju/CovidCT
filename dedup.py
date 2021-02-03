import os
import pandas as pd

dirname = os.path.dirname(__file__)

file_name = os.path.join(dirname, 'csv-small/all.csv')
file_name_output = os.path.join(dirname, 'csv-small/all-dedup.csv')

df = pd.read_csv(file_name, sep=',')
df.drop_duplicates(subset=None, inplace=True)

df.to_csv(file_name_output, index=False)