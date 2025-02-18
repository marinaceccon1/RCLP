import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms
from src.data.dataset import NIH
from src.data.dataset import CheXpert
import numpy as np
import random
import pandas as pd
import os

def read_indices_from_txt(file_path):
    indices = []
    with open(file_path, 'r') as file:
        for line in file:
            index = int(line.strip())
            indices.append(index)
    return indices

def save_indices_to_txt(indices, file_path):
    with open(file_path, 'w') as file:
        for index in indices:
            file.write(str(index) + '\n')

# Define a function to prioritize "frontal" images
def prioritize_frontal(group):
    frontal_indices = group[group['Path'].str.contains('frontal')].index.tolist()
    return frontal_indices if frontal_indices else group.index.tolist()

def filter_target(target, task_labels, device):
    new_target = torch.zeros((target.size(0),len(task_labels))).to(device)
    for i in range(target.size(0)):
        for j in range(len(task_labels)):
            new_target[i][j] = target[i][task_labels[j]]
    return new_target

class MergedDataset(Dataset):
    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.total_size = len(dataset1) + len(dataset2)
        self.indices = list(range(self.total_size))
        random.shuffle(self.indices)  # Shuffle the indices

    def __getitem__(self, idx):
        idx = self.indices[idx]  # Use shuffled index
        if idx < len(self.dataset1):
            return self.dataset1[idx]
        else:
            idx -= len(self.dataset1)
            return self.dataset2[idx]

    def __len__(self):
        return self.total_size
    
def filter_single_target(target, task_labels, device):
    new_target = torch.zeros(len(task_labels)).to(device)
    for j in range(len(task_labels)):
        new_target[j] = target[task_labels[j]]
    return new_target


def import_nih_dfs(base_path):
    #read the csv file relative to the NIH dataset
    path_to_csv = os.path.join(base_path, "dataset/d_nih_aug.csv")
    d_nih_aug_csv = pd.read_csv(path_to_csv)
    
    #read the indices associated to the train, validtion and test set from the txt files
    train_indices_file_nih = os.path.join(base_path, 'indices/train_indices_nih.txt')
    val_indices_file_nih = os.path.join(base_path, 'indices/val_indices_nih.txt')
    test_indices_file_nih = os.path.join(base_path, 'indices/test_indices_nih.txt')

    train_indices_list_nih = read_indices_from_txt(train_indices_file_nih)
    val_indices_list_nih = read_indices_from_txt(val_indices_file_nih)
    test_indices_list_nih = read_indices_from_txt(test_indices_file_nih)
    
    #Create the corresponding dataframes
    train_df_nih = d_nih_aug_csv[d_nih_aug_csv["index"].isin(train_indices_list_nih)].reset_index(drop=True)
    val_df_nih = d_nih_aug_csv[d_nih_aug_csv["index"].isin(val_indices_list_nih)].reset_index(drop=True)
    test_df_nih = d_nih_aug_csv[d_nih_aug_csv["index"].isin(test_indices_list_nih)].reset_index(drop=True)
    
    return train_df_nih, val_df_nih, test_df_nih

def import_cxp_dfs(base_path):
    ##read the csv files relative to the CXP datasets
    cxp_csv_train_file = os.path.join(base_path, "dataset/chexpertchestxrays-u20210408/train.csv")
    cxp_csv_val_file = os.path.join(base_path, "dataset/chexpertchestxrays-u20210408/valid.csv")

    cxp_train_csv = pd.read_csv(cxp_csv_train_file)
    cxp_val_csv = pd.read_csv(cxp_csv_val_file)
    
    #create a new column with the patient ID (extracted from the path)
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
    
    #Read the train, validation and test indices from the corresponding txt files
    train_patient_indices_cxp = read_indices_from_txt(os.path.join(base_path,'indices/train_indices_cxp.txt'))
    val_patient_indices_cxp = read_indices_from_txt(os.path.join(base_path,'indices/val_indices_cxp.txt'))
    test_patient_indices_cxp = read_indices_from_txt(os.path.join(base_path,'indices/test_indices_cxp.txt'))
    
    train_df_cxp = cxp_df[cxp_df.index.isin(train_patient_indices_cxp)].reset_index(drop=True)
    val_df_cxp = cxp_df[cxp_df.index.isin(val_patient_indices_cxp)].reset_index(drop=True)
    test_df_cxp = cxp_df[cxp_df.index.isin(test_patient_indices_cxp)].reset_index(drop=True)
    
    return train_df_cxp, val_df_cxp, test_df_cxp

def create_datasets(dfs_cxp, dfs_nih, train_indices_tasks, val_indices_tasks, test_indices_tasks, tasks_labels_cxp, tasks_labels_nih, path_image_cxp, train_transform_cxp, val_transform_cxp, test_transform_cxp, path_image_nih, train_transform_nih, val_transform_nih, test_transform_nih):
    #create dataframes from the lists of indices
    train_df_cxp = dfs_cxp[0]
    val_df_cxp = dfs_cxp[1]
    test_df_cxp = dfs_cxp[2]
    
    train_df_nih = dfs_nih[0]
    val_df_nih = dfs_nih[1]
    test_df_nih = dfs_nih[2]
    
    train_dfs_cxp = [train_df_cxp[train_df_cxp.index.isin(train_indices_tasks[j])] for j in tasks_labels_cxp]
    train_dfs_nih = [train_df_nih[train_df_nih.index.isin(train_indices_tasks[j])] for j in tasks_labels_nih]

    val_dfs_cxp = [val_df_cxp[val_df_cxp.index.isin(val_indices_tasks[j])] for j in tasks_labels_cxp]
    val_dfs_nih = [val_df_nih[val_df_nih.index.isin(val_indices_tasks[j])] for j in tasks_labels_nih]

    test_dfs_cxp = [test_df_cxp[test_df_cxp.index.isin(test_indices_tasks[j])] for j in tasks_labels_cxp]
    test_dfs_nih = [test_df_nih[test_df_nih.index.isin(test_indices_tasks[j])] for j in tasks_labels_nih]
    
    #create datasets from the dataframes
    train_datasets_cxp = [CheXpert(train_df_cxp, path_image=path_image_cxp, transform = train_transform_cxp) for train_df_cxp in train_dfs_cxp]
    train_datasets_nih = [NIH(train_df_nih, path_image=path_image_nih, transform=train_transform_nih) for train_df_nih in train_dfs_nih]

    val_datasets_cxp = [CheXpert(val_df_cxp, path_image=path_image_cxp, transform = val_transform_cxp) for val_df_cxp in val_dfs_cxp]
    val_datasets_nih = [NIH(val_df_nih, path_image=path_image_nih, transform=val_transform_nih) for val_df_nih in val_dfs_nih]

    test_datasets_cxp = [CheXpert(test_df_cxp, path_image=path_image_cxp, transform = test_transform_cxp) for test_df_cxp in test_dfs_cxp]
    test_datasets_nih = [NIH(test_df_nih, path_image=path_image_nih, transform=test_transform_nih) for test_df_nih in test_dfs_nih]
    
    #create lists of datasets alternating cxp and nih datasets
    train_datasets = [train_datasets_cxp[0], train_datasets_nih[0], train_datasets_cxp[1], train_datasets_cxp[2], 
                      train_datasets_nih[1], train_datasets_nih[2], train_datasets_nih[3]]
    val_datasets = [val_datasets_cxp[0], val_datasets_nih[0], val_datasets_cxp[1], val_datasets_cxp[2], 
                      val_datasets_nih[1], val_datasets_nih[2], val_datasets_nih[3]]
    test_datasets = [test_datasets_cxp[0], test_datasets_nih[0], test_datasets_cxp[1], test_datasets_cxp[2], 
                      test_datasets_nih[1], test_datasets_nih[2], test_datasets_nih[3]]
    
    return train_datasets, val_datasets, test_datasets


def create_dataloaders(train_datasets, val_datasets, test_datasets, tasks_labels_cxp, tasks_labels_nih):
    #create dataloaders from the datasets
    train_datasets_cxp = [train_datasets[i] for i in tasks_labels_cxp]
    val_datasets_cxp = [val_datasets[i] for i in tasks_labels_cxp]
    test_datasets_cxp = [test_datasets[i] for i in tasks_labels_cxp]
    
    train_datasets_nih = [train_datasets[i] for i in tasks_labels_nih]
    val_datasets_nih = [val_datasets[i] for i in tasks_labels_nih]
    test_datasets_nih = [test_datasets[i] for i in tasks_labels_nih]
    
    train_dataloaders_cxp = [DataLoader(train_dataset, batch_size=48, shuffle=True, num_workers=12, pin_memory=True) 
                        for train_dataset in train_datasets_cxp]
    val_dataloaders_cxp = [DataLoader(val_dataset, batch_size=48, shuffle=True,num_workers=12, pin_memory=True) 
                        for val_dataset in val_datasets_cxp]
    test_dataloaders_cxp = [DataLoader(test_dataset, batch_size=48, shuffle=True, num_workers = 12, pin_memory = True) 
                        for test_dataset in test_datasets_cxp]
    
    #create dataloaders from the datasets
    train_dataloaders_nih = [DataLoader(train_dataset, batch_size=32, shuffle=True,num_workers=12, pin_memory=True) 
                        for train_dataset in train_datasets_nih]
    val_dataloaders_nih = [DataLoader(val_dataset, batch_size=32, shuffle=True,num_workers=12, pin_memory=True) 
                        for val_dataset in val_datasets_nih]
    test_dataloaders_nih = [DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers = 12, pin_memory = True) 
                        for test_dataset in test_datasets_nih]
    
    #create lists of dataloaders alternating cxp and nih dataloaders
    train_dataloaders = [train_dataloaders_cxp[0], train_dataloaders_nih[0], train_dataloaders_cxp[1], 
                         train_dataloaders_cxp[2], train_dataloaders_nih[1], train_dataloaders_nih[2], 
                         train_dataloaders_nih[3]]
    val_dataloaders = [val_dataloaders_cxp[0], val_dataloaders_nih[0], val_dataloaders_cxp[1], 
                         val_dataloaders_cxp[2], val_dataloaders_nih[1], val_dataloaders_nih[2], 
                         val_dataloaders_nih[3]]
    test_dataloaders = [test_dataloaders_cxp[0], test_dataloaders_nih[0], test_dataloaders_cxp[1], 
                         test_dataloaders_cxp[2], test_dataloaders_nih[1], test_dataloaders_nih[2], 
                         test_dataloaders_nih[3]]
    
    return train_dataloaders, val_dataloaders, test_dataloaders

def create_datasets_joint_cxp(train_indices_tasks, val_indices_tasks, test_indices_tasks, train_df_cxp, val_df_cxp, test_df_cxp, path_image_cxp, train_transform_cxp, val_transform_cxp, test_transform_cxp, tasks_labels_cxp):
    #each list contains all the indices in all the cxp tasks/nih tasks
    train_indices_joint_cxp = []

    val_indices_joint_cxp = []

    test_indices_joint_cxp = []

    for i in tasks_labels_cxp:
        for j in train_indices_tasks[i]:
            if j not in train_indices_joint_cxp:
                train_indices_joint_cxp.append(j)

    for i in tasks_labels_cxp:
        for j in val_indices_tasks[i]:
            if j not in val_indices_joint_cxp:
                val_indices_joint_cxp.append(j)

    for i in tasks_labels_cxp:
        for j in test_indices_tasks[i]:
            if j not in test_indices_joint_cxp:
                test_indices_joint_cxp.append(j)
                
    #build the dataframes from the lists of indices
    train_df_joint_cxp = train_df_cxp[train_df_cxp.index.isin(train_indices_joint_cxp)]
    val_df_joint_cxp = val_df_cxp[val_df_cxp.index.isin(val_indices_joint_cxp)]
    test_df_joint_cxp = test_df_cxp[test_df_cxp.index.isin(test_indices_joint_cxp)]
    
    #build the datasets from the dataframes
    train_dataset_joint_cxp = CheXpert(train_df_joint_cxp, path_image=path_image_cxp, transform = train_transform_cxp)
    val_dataset_joint_cxp = CheXpert(val_df_joint_cxp,path_image=path_image_cxp, transform = val_transform_cxp)
    test_dataset_joint_cxp = CheXpert(test_df_joint_cxp, path_image=path_image_cxp, transform = test_transform_cxp)
    
    path_list_train = train_df_joint_cxp["Path"].tolist()
    path_list_val = val_df_joint_cxp["Path"].tolist()
    
    return train_dataset_joint_cxp, val_dataset_joint_cxp, test_dataset_joint_cxp, path_list_train, path_list_val

def create_datasets_joint_nih(train_indices_tasks, val_indices_tasks, test_indices_tasks, train_df_nih, val_df_nih, test_df_nih, path_image_nih, train_transform_nih, val_transform_nih, test_transform_nih, tasks_labels_nih):
    #each list contains all the indices in all the cxp tasks/nih tasks
    train_indices_joint_nih = []

    val_indices_joint_nih = []

    test_indices_joint_nih = []

    for i in tasks_labels_nih:
        for j in train_indices_tasks[i]:
            if j not in train_indices_joint_nih:
                train_indices_joint_nih.append(j)

    for i in tasks_labels_nih:
        for j in val_indices_tasks[i]:
            if j not in val_indices_joint_nih:
                val_indices_joint_nih.append(j)

    for i in tasks_labels_nih:
        for j in test_indices_tasks[i]:
            if j not in test_indices_joint_nih:
                test_indices_joint_nih.append(j)
                
    #build the dataframes from the lists of indices
    train_df_joint_nih = train_df_nih[train_df_nih.index.isin(train_indices_joint_nih)]
    val_df_joint_nih = val_df_nih[val_df_nih.index.isin(val_indices_joint_nih)]
    test_df_joint_nih = test_df_nih[test_df_nih.index.isin(test_indices_joint_nih)]
    
    #build the datasets from the dataframes
    train_dataset_joint_nih = NIH(train_df_joint_nih, path_image=path_image_nih, transform = train_transform_nih)
    val_dataset_joint_nih = NIH(val_df_joint_nih,path_image=path_image_nih, transform = val_transform_nih)
    test_dataset_joint_nih = NIH(test_df_joint_nih, path_image=path_image_nih, transform = test_transform_nih)
    
    return train_dataset_joint_nih, val_dataset_joint_nih, test_dataset_joint_nih

def modify_dataset_labels(dataset, task_labels, batch_size=32, num_workers=12, pin_memory=True):
    modified_dataset = []
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)

    for batch in dataloader:
        imgs, labels, paths = batch
        for img, label, path in zip(imgs, labels, paths):
            new_label = torch.zeros_like(label)
            for i in task_labels:
                new_label[i] = label[i]
            modified_dataset.append((img, new_label, path))
    return modified_dataset

def create_buffer(train_datasets, tasks_labels):
    #Total size of the replay buffer (3% of the total size of the training stream)
    for i in range(len(tasks_labels)):
        modified_train_dataset = modify_dataset_labels(train_datasets[i], tasks_labels[i])
    train_datasets[i] = modified_train_dataset
    
    total_length = sum(len(train_dataset) for train_dataset in train_datasets)
    subset_size = int(total_length * 3 / 100)

    replayed_datasets = []
    for i in range(len(train_datasets)):
        #the sublist to append in the list
        subset_datasets = []
        if i > 0:
            new_subset_size = int(subset_size/(i))
            #sample the subset from the original dataset
            indices = list(range(len(train_datasets[i-1])))
            random_indices = random.sample(indices, new_subset_size) 
            random_subset_dataset = Subset(train_datasets[i-1], random_indices)

            #append the subset to the buffer
            subset_datasets.append(random_subset_dataset)

            for j in range(len(replayed_datasets[i-1])):       
                #sample the subset from the original dataset
                indices = list(range(len(replayed_datasets[i-1][j])))
                random_indices = random.sample(indices, new_subset_size)
                random_subset_dataset = Subset(replayed_datasets[i-1][j], random_indices)

                #append the subset to the buffer
                subset_datasets.append(random_subset_dataset)

        #append the buffer to the list of buffers
        replayed_datasets.append(subset_datasets)
        
    new_replayed_datasets = []

    for j in range(1,len(replayed_datasets)):
        datasets_list = replayed_datasets[j]
        merged_dataset = datasets_list[0]  # Initialize with the first dataset
        for dataset in datasets_list[1:]:
            merged_dataset = MergedDataset(merged_dataset, dataset)  # Merge with each subsequent dataset
        new_replayed_datasets.append(merged_dataset)
    return new_replayed_datasets
