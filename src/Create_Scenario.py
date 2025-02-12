import sys
import os

current_path = os.path.abspath(__file__)
base_path = os.path.dirname(os.path.dirname(current_path))
sys.path.append(base_path)


from sklearn.metrics import roc_auc_score
import torch
import torchxrayvision as xrv
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
import matplotlib.pyplot as plt
from src.data.dataset import NIH
from src.data.dataset import CheXpert
from src.data.utils import read_indices_from_txt
from src.data.utils import prioritize_frontal
from sklearn.metrics import roc_auc_score
import time
import os
from PIL import Image
import pandas as pd
import random

#read the csv file of the NIH dataset. Needed to define the NIH dataframe
path_to_csv = os.path.join(base_path, "dataset/d_nih_aug.csv")
d_nih_aug_csv = pd.read_csv(path_to_csv)

#IF THE FILES "train_nih.txt", "val_nih.txt" and "test_nih.txt" HAVE ALREADY BEEN CREATED, IT'S NOT NEEDED TO COMPUTE THE LINES UP TO 139
# create a list of indices for all patients in d_nih
all_indices = list(range(np.max(d_nih_aug_csv["Patient ID"])))

# shuffle the list of indices
random.shuffle(all_indices)

# calculate the number of images in each dataset
num_train = int(np.max(d_nih_aug_csv["Patient ID"]) * 0.8)
num_val_test = np.max(d_nih_aug_csv["Patient ID"]) - num_train
num_val = num_val_test // 2
num_test = num_val_test - num_val

# split the indices into three sets
#"We use a 80-10-10 train-validation-test split with no patient shared across splits."
train_patient_indices_nih = all_indices[:num_train]
val_patient_indices_nih = all_indices[num_train:num_train+num_val]
test_patient_indices_nih = all_indices[num_train+num_val:]

# verify that the lengths add up correctly
assert len(train_patient_indices_nih) == num_train
assert len(val_patient_indices_nih) == num_val
assert len(test_patient_indices_nih) == num_test
assert len(train_patient_indices_nih) + len(val_patient_indices_nih) + len(test_patient_indices_nih) == np.max(d_nih_aug_csv["Patient ID"])

#convert the lists into sets to verify that the sets have no element in common
train_set_nih = set(train_patient_indices_nih)
val_set_nih = set(val_patient_indices_nih)
test_set_nih = set(test_patient_indices_nih)

if len(train_set_nih.intersection(val_set_nih)) > 0:
    print("train and val sets have common indices")
if len(train_set_nih.intersection(test_set_nih)) > 0:
    print("train and test sets have common indices")
if len(val_set_nih.intersection(test_set_nih)) > 0:
    print("val and test sets have common indices")

#train_indices is the list of indices of images in the training set, it contains the indices of all images whose patient index
#is in train_patient_indices
train_indices_nih = {}
#val_indices is the list of indices of images in the validation set it contains the indices of all images whose patient index
#is in val_patient_indices
val_indices_nih = {}
#test_indices is the list of indices of images in the test set, it contains the indices of all images whose patient index
#is in test_patient_indices
test_indices_nih = {}

for i in d_nih_aug_csv["Patient ID"]:
    train_indices_nih[i] = []
    val_indices_nih[i] = []
    test_indices_nih[i] = []

for i in range(len(d_nih_aug_csv)):
    if d_nih_aug_csv["Patient ID"][i] in train_patient_indices_nih:
        train_indices_nih[d_nih_aug_csv["Patient ID"][i]].append(i)
    elif d_nih_aug_csv["Patient ID"][i] in val_patient_indices_nih:
        val_indices_nih[d_nih_aug_csv["Patient ID"][i]].append(i)
    elif d_nih_aug_csv["Patient ID"][i] in test_patient_indices_nih:
        test_indices_nih[d_nih_aug_csv["Patient ID"][i]].append(i)

for i in train_indices_nih.keys():
    if (len(train_indices_nih[i])>1):
        for idx in train_indices_nih[i]:
            if d_nih_aug_csv["Finding Labels"][idx] == "No Finding":
                train_indices_nih[i].remove(idx)

for i in val_indices_nih.keys():
    if (len(val_indices_nih[i])>1):
        for idx in val_indices_nih[i]:
            if d_nih_aug_csv["Finding Labels"][idx] == "No Finding":
                val_indices_nih[i].remove(idx)

for i in test_indices_nih.keys():
    if (len(test_indices_nih[i])>1):
        for idx in test_indices_nih[i]:
            if d_nih_aug_csv["Finding Labels"][idx] == "No Finding":
                test_indices_nih[i].remove(idx)

train_indices_list_nih = []

for i in train_indices_nih.keys():
    if (len(train_indices_nih[i])>0):
        random_number = random.randint(0, len(train_indices_nih[i])-1)
        train_indices_list_nih.append(train_indices_nih[i][random_number])

val_indices_list_nih = []

for i in val_indices_nih.keys():
    if (len(val_indices_nih[i])>0):
        random_number = random.randint(0, len(val_indices_nih[i])-1)
        val_indices_list_nih.append(val_indices_nih[i][random_number])

test_indices_list_nih = []

for i in test_indices_nih.keys():
    if (len(test_indices_nih[i])>0):
        random_number = random.randint(0, len(test_indices_nih[i])-1)
        test_indices_list_nih.append(test_indices_nih[i][random_number])

train_indices_file_nih = '/home/marina/NewContinual/indices/train_nih.txt'
val_indices_file_nih = '/home/marina/NewContinual/indices/val_nih.txt'
test_indices_file_nih = '/home/marina/NewContinual/indices/test_nih.txt'

save_indices_to_txt(train_indices_list_nih, train_indices_file_nih)
save_indices_to_txt(val_indices_list_nih, val_indices_file_nih)
save_indices_to_txt(test_indices_list_nih, test_indices_file_nih)

#IF THE CODE ABOVE HAS ALREADY BEEN COMPUTED, START FROM HERE
train_indices_file_nih = '/home/marina/NewContinual/indices/train_nih.txt'
val_indices_file_nih = '/home/marina/NewContinual/indices/val_nih.txt'
test_indices_file_nih = '/home/marina/NewContinual/indices/test_nih.txt'

train_indices_list_nih = read_indices_from_txt(train_indices_file_nih)
val_indices_list_nih = read_indices_from_txt(val_indices_file_nih)
test_indices_list_nih = read_indices_from_txt(test_indices_file_nih)

#define the dtaframes relative to the train, validation and test set
train_df_nih = d_nih_aug_csv[d_nih_aug_csv.index.isin(train_indices_list_nih)].reset_index(drop=True)
val_df_nih = d_nih_aug_csv[d_nih_aug_csv["index"].isin(val_indices_list_nih)].reset_index(drop=True)
test_df_nih = d_nih_aug_csv[d_nih_aug_csv["index"].isin(test_indices_list_nih)].reset_index(drop=True)

#IF "train_cxp.txt", "val_cxp.txt" and "test_cxp.txt" ALREADY EXIST, DON'T NEED TO RUN THIS CODE, CAN SKIP TO LINE 233
#define the paths where the train and validation csv files can be found
cxp_csv_train_file = os.path.join(base_path,"dataset/chexpertchestxrays-u20210408/train.csv")
cxp_csv_val_file = os.path.join(base_path,"dataset/chexpertchestxrays-u20210408/valid.csv")

cxp_train_csv = pd.read_csv(cxp_csv_train_file)
cxp_val_csv = pd.read_csv(cxp_csv_val_file)

#consider only the patient ID in the path
cxp_train_csv['patient_id'] = cxp_train_csv['Path'].apply(lambda x: x.split('/')[2])
cxp_val_csv['patient_id'] = cxp_val_csv['Path'].apply(lambda x: x.split('/')[2])

# Associate to each patient ID  list containing all indices of the dataframe that correspond to that patient ID
# Consider only the indices such that the path contains the word "frontal"
frontal_patient_indices_train = cxp_train_csv.groupby('patient_id', group_keys=False).apply(prioritize_frontal)
frontal_patient_indices_val = cxp_val_csv.groupby('patient_id', group_keys=False).apply(prioritize_frontal)

#Create a list out of all the indices
frontal_patient_indices_train = [item for sublist in frontal_patient_indices_train for item in sublist]
frontal_patient_indices_val = [item for sublist in frontal_patient_indices_val for item in sublist]

#create the corresponding dataframes
frontal_df_train = cxp_train_csv.loc[frontal_patient_indices_train]
frontal_df_val = cxp_val_csv.loc[frontal_patient_indices_val]

# Keep only one image per patient
unique_patients_df_train = frontal_df_train.drop_duplicates(subset='patient_id', keep='first')
unique_patients_df_val = frontal_df_val.drop_duplicates(subset='patient_id', keep='first')

# Get the indices of the unique patients
unique_patient_indices_train = unique_patients_df_train.index.tolist()
unique_patient_indices_val = unique_patients_df_val.index.tolist()

cxp_train_df =frontal_df_train.loc[unique_patient_indices_train]
cxp_val_df =frontal_df_val.loc[unique_patient_indices_val]

#change the paths so that they correspond to the actual location of the dataset
new_prefix = "train"
cxp_train_df['Path'] = cxp_train_df['Path'].str.replace('^CheXpert-v1\.0/train', new_prefix, regex=True)
cxp_val_df['Path'] = cxp_val_df['Path'].str.replace('^CheXpert-v1\.0/valid', new_prefix, regex=True)

# Change the name of 'Pleural Effusion' in 'Effusion' so that it matches NIH
cxp_val_df.rename(columns={'Pleural Effusion': 'Effusion'}, inplace=True)
cxp_train_df.rename(columns={'Pleural Effusion': 'Effusion'}, inplace=True)

# Concatenate vertically
cxp_df = pd.concat([cxp_train_df, cxp_val_df], axis=0).reset_index(drop = True)

train_df_cxp = cxp_df[cxp_df.index.isin(train_patient_indices_cxp)].reset_index(drop=True)
val_df_cxp = cxp_df[cxp_df.index.isin(val_patient_indices_cxp)].reset_index(drop=True)
test_df_cxp = cxp_df[cxp_df.index.isin(test_patient_indices_cxp)].reset_index(drop=True)
# create a list of indices for all patients in d_nih
all_indices_cxp = list(range(len(cxp_df)))

# shuffle the list of indices
random.shuffle(all_indices_cxp)

# calculate the number of images in each dataset
num_train_cxp = int(len(cxp_df) * 0.8)
num_val_test_cxp = len(cxp_df) - num_train_cxp
num_val_cxp = num_val_test_cxp // 2
num_test_cxp = num_val_test_cxp - num_val_cxp

# split the indices into three sets
#"We use a 80-10-10 train-validation-test split with no patient shared across splits."
train_patient_indices_cxp = all_indices_cxp[:num_train_cxp]
val_patient_indices_cxp = all_indices_cxp[num_train_cxp:num_train_cxp+num_val_cxp]
test_patient_indices_cxp = all_indices_cxp[num_train_cxp+num_val_cxp:]

# verify that the lengths add up correctly
assert len(train_patient_indices_cxp) == num_train_cxp
assert len(val_patient_indices_cxp) == num_val_cxp
assert len(test_patient_indices_cxp) == num_test_cxp
assert len(train_patient_indices_cxp) + len(val_patient_indices_cxp) + len(test_patient_indices_cxp) == len(cxp_df)

train_indices_file_cxp = os.path.join(base_path, '/indices/train_cxp.txt')
val_indices_file_cxp = os.path.join(base_path, '/indices/val_cxp.txt')
test_indices_file_cxp = os.path.join(base_path, '/indices/test_cxp.txt')

save_indices_to_txt(train_patient_indices_cxp, train_indices_file_cxp)
save_indices_to_txt(val_patient_indices_cxp, val_indices_file_cxp)
save_indices_to_txt(test_patient_indices_cxp, test_indices_file_cxp)

#IF THE CODE ABOVE HAS ALREADY BEEN COMPUTED, START FROM HERE
train_indices_file_cxp = os.path.join(base_path, '/indices/train_cxp.txt')
val_indices_file_cxp = os.path.join(base_path, '/indices/val_cxp.txt')
test_indices_file_cxp = os.path.join(base_path, '/indices/test_cxp.txt')

train_patient_indices_cxp = read_indices_from_txt(train_indices_file_CXP)
val_patient_indices_cxp = read_indices_from_txt(val_indices_file_cxp)
test_patient_indices_cxp = read_indices_from_txt(test_indices_file_cxp)

train_df_cxp = cxp_df[cxp_df.index.isin(train_patient_indices_cxp)].reset_index(drop=True)
val_df_cxp = cxp_df[cxp_df.index.isin(val_patient_indices_cxp)].reset_index(drop=True)
test_df_cxp = cxp_df[cxp_df.index.isin(test_patient_indices_cxp)].reset_index(drop=True)

pathologies = ['Lung Opacity',
              'Atelectasis',
              'Cardiomegaly',
              'Consolidation',
              'Edema',
              'Effusion',
              'Emphysema',
              'Enlarged Cardiomediastinum',
              'Fibrosis',
              'Fracture',
              'Hernia',
              'Infiltration',
              'Lung Lesion',
              'Mass',
              'Nodule',
              'Pleural_Thickening',
              'Pleural Other',
              'Pneumonia',
              'Pneumothorax']

#pathologies from the list associted to the CXP dataset
reference_vector_cxp = [0, 1, 2, 3, 4, 5, 7, 9, 12, 16, 17, 18]
#pathologies from the list associted to the NIH dataset
reference_vector_nih = [1, 2, 3, 4, 5, 6, 8, 10, 11, 13, 14, 15, 17, 18]

nih_pathologies = [pathologies[reference_vector_nih[i]] for i in range(len(reference_vector_nih))]
cxp_pathologies = [pathologies[reference_vector_cxp[i]] for i in range(len(reference_vector_cxp))]

#tasks_labels: [3,17,18] from CXP, [3,17,18] from NIH, [0,7,9,12,16] from CXP, [1,2,4,5] from CXP, [1,2,4,5] from NIH
# [6,8,10,11] from NIH, [13,14,15] from NIH

#domain incremental, class incremental, class incremental, class incremental, domain incremental, 
#class incremental, class incremental
tasks_labels = [[3,17,18],[3,17,18],[0,7,9,12,16],[1,2,4,5],[1,2,4,5],[6,8,10,11],[13,14,15]]

#lists of lists: each list will contain the indices of the samples associated to that task
train_indices_tasks= [[],[],[],[],[],[],[]]
val_indices_tasks= [[],[],[],[],[],[],[]]
test_indices_tasks= [[],[],[],[],[],[],[]]

#indices of the tasks in tasks_labels associated to CXP
tasks_labels_cxp = [0,2,3]
#indices of the tasks in tasks_labels associated to NIH
tasks_labels_nih = [1,4,5,6]

#Consider all samples in the training dataset
for i in range(len(train_df_cxp)):
    #Consider all pathologies of cxp
    for j in range(len(cxp_pathologies)):
        #if the patient has the corresponding pathology
        if(train_df_cxp[cxp_pathologies[j].strip()].iloc[i].astype('float') > 0):
            #find the task that contains that pathology and add the sample to the train_indices_tasks of that task
            #we need to consider only tasks_labels[k] and train_indices_tasks[k] for k contained in tasks_labels_cxp
            for k in tasks_labels_cxp:
                if reference_vector_cxp[j] in tasks_labels[k] and i not in train_indices_tasks[k]:
                    train_indices_tasks[k].append(i)
                    
for i in range(len(val_df_cxp)):
    for j in range(len(cxp_pathologies)):
        if(val_df_cxp[cxp_pathologies[j].strip()].iloc[i].astype('float') > 0):
            for k in tasks_labels_cxp:
                if reference_vector_cxp[j] in tasks_labels[k] and i not in val_indices_tasks[k]:
                    val_indices_tasks[k].append(i)
                    
for i in range(len(test_df_cxp)):
    for j in range(len(cxp_pathologies)):
        if(test_df_cxp[cxp_pathologies[j].strip()].iloc[i].astype('float') > 0):
            for k in tasks_labels_cxp:
                if reference_vector_cxp[j] in tasks_labels[k] and i not in test_indices_tasks[k]:
                    test_indices_tasks[k].append(i)

for i in range(len(train_df_nih)):
    for j in range(len(nih_pathologies)):
        if (nih_pathologies[j].strip() in train_df_nih["Finding Labels"].iloc[i]):
            for k in tasks_labels_nih:
                if reference_vector_nih[j] in tasks_labels[k] and i not in train_indices_tasks[k]:
                    train_indices_tasks[k].append(i)
                    
for i in range(len(val_df_nih)):
    for j in range(len(nih_pathologies)):
        if (nih_pathologies[j].strip() in val_df_nih["Finding Labels"].iloc[i]):
            for k in tasks_labels_nih:
                if reference_vector_nih[j] in tasks_labels[k] and i not in val_indices_tasks[k]:
                    val_indices_tasks[k].append(i)
                    
for i in range(len(test_df_nih)):
    for j in range(len(nih_pathologies)):
        if (nih_pathologies[j].strip() in test_df_nih["Finding Labels"].iloc[i]):
            for k in tasks_labels_nih:
                if reference_vector_nih[j] in tasks_labels[k] and i not in test_indices_tasks[k]:
                    test_indices_tasks[k].append(i)

# Define the file path
file_path = os.path.join(base_path, "indices/train_indices_tasks.txt")

# Open the file for writing
with open(file_path, "w") as f:
    # Iterate over each sublist in train_indices_tasks
    for sublist in train_indices_tasks:
        # Convert the sublist to a string and write it to the file
        f.write(" ".join(map(str, sublist)) + "\n")

# Define the file path
file_path = os.path.join(base_path,"indices/val_indices_tasks.txt")

# Open the file for writing
with open(file_path, "w") as f:
    # Iterate over each sublist in train_indices_tasks
    for sublist in val_indices_tasks:
        # Convert the sublist to a string and write it to the file
        f.write(" ".join(map(str, sublist)) + "\n")

# Define the file path
file_path = os.path.join(base_path,"indices/test_indices_tasks.txt")

# Open the file for writing
with open(file_path, "w") as f:
    # Iterate over each sublist in train_indices_tasks
    for sublist in test_indices_tasks:
        # Convert the sublist to a string and write it to the file
        f.write(" ".join(map(str, sublist)) + "\n")
