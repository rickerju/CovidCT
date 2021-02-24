import os
import shutil

import math
import pandas as pd

data_directory = 'data/'
fold_csv_directory = 'fold-csv/'
fold_data_directory = 'fold-data/'

relative_path = os.path.dirname(__file__)
data_path = os.path.join(relative_path, data_directory)
fold_csv_path = os.path.join(relative_path, fold_csv_directory)
fold_data_path = os.path.join(relative_path, fold_data_directory)

training_set_size = 4000
validation_set_size = 500
test_set_size = 5000

df = pd.read_csv(r"all.csv")

df_normal = df[df['class'] == 'normal']
df_covid = df[df['class'] == 'covid']

# training_set should be 50/50
df_normal_chosen = df_normal.sample(n=math.floor(training_set_size / 2))
df_covid_chosen = df_covid.sample(n=math.floor(training_set_size / 2))
training_set = pd.concat([
    df_normal_chosen,
    df_covid_chosen
], sort=False)
df_normal = df_normal.drop(df_normal_chosen.index)
df_covid = df_covid.drop(df_covid_chosen.index)

df_unused = df.drop(training_set.index)

# validation_set should be 50/50
validation_set = pd.concat([
    df_normal.sample(n=math.floor(validation_set_size / 2)),
    df_covid.sample(n=math.floor(validation_set_size / 2))
])

df_unused = df_unused.drop(validation_set.index)

# test set should be a sample of all records left
test_set = df_unused.sample(n=test_set_size)

# make fold path directory if not exists
if not os.path.isdir(fold_csv_path):
    os.mkdir(fold_csv_path)

# write files
training_set.to_csv(fold_csv_path + 'train.csv', index=False)
validation_set.to_csv(fold_csv_path + 'validation.csv', index=False)
test_set.to_csv(fold_csv_path + 'test.csv', index=False)

# generate training set folds
# index = 0
# training_set_chunks = chunked(training_set.index, 10)
# for i in training_set_chunks:
#     chunk = df.iloc[i]
#     chunk.to_csv(fold_csv_path + 'train' + str(index) + '.csv', index=False)
#     index = index + 1
#
# # generate validation set folds
# index = 0
# validation_set_chunks = chunked(validation_set.index, 10)
# for i in validation_set_chunks:
#     chunk = df.iloc[i]
#     chunk.to_csv(fold_csv_path + 'validation' + str(index) + '.csv', index=False)
#     index = index + 1
#
# # generate test set folds
# index = 0
# test_set_chunks = chunked(test_set.index, 20)
# for i in test_set_chunks:
#     chunk = df.iloc[i]
#     chunk.to_csv(fold_csv_path + 'test' + str(index) + '.csv', index=False)
#     index = index + 1

# get a dataframe of all used files
df_used = pd.concat([
    training_set,
    validation_set,
    test_set
])

# make fold data path directory if not exists
if not os.path.isdir(fold_data_path):
    os.mkdir(fold_data_path)

# move all used files to a new directory
for record in df_used.itertuples(index=False):
    filename = record[0]
    shutil.copy(data_path + filename, fold_data_path + filename)