#Import the necessary libraries

import sys
import os
base_path = #insert the path where the repository was imported
sys.path.append(base_path)

from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from src.data.dataset import NIH
from src.data.dataset import CheXpert
from src.data.utils import import_nih_dfs
from src.data.utils import import_cxp_dfs
from src.data.utils import create_datasets
from src.data.utils import create_datasets_joint_cxp
from src.data.utils import create_datasets_joint_nih
from src.data.utils import create_dataloaders
from src.eval.utils import eval_model_naive
from src.eval.utils import compute_auc_and_f1
from src.training.utils import train_model_naive
import time
import os

#define the gpu device used to attach the tensors
device = torch.device('cuda:2')

#Create the dataframes corresponding to the training, validation and test set of NIH
train_df_nih, val_df_nih, test_df_nih = import_nih_dfs(base_path)
dfs_nih = [train_df_nih, val_df_nih, test_df_nih]

#path where the NIH images can be found
path_image_nih = os.path.join(base_path,"dataset/archive/")

#define the transforms associated to the NIH dataset
normalize_nih = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

train_transform_nih = transforms.Compose([transforms.RandomHorizontalFlip(),
                                transforms.RandomRotation(10),
                                transforms.Resize((256,256)),
                                transforms.CenterCrop(256),
                                transforms.ToTensor(),
                                normalize_nih])
val_transform_nih = transforms.Compose([transforms.Resize((256,256)),
                                     transforms.CenterCrop(256),
                                     transforms.ToTensor(),
                                     normalize_nih])
test_transform_nih = transforms.Compose([transforms.Resize((256,256)),
                                     transforms.CenterCrop(256),
                                     transforms.ToTensor(),
                                     normalize_nih])

#Create the dataframes corresponding to the training, validation and test set of NIH
train_df_cxp, val_df_cxp, test_df_cxp = import_cxp_dfs(base_path)
dfs_cxp = [train_df_cxp, val_df_cxp, test_df_cxp]

#path of the dataset
path_image_cxp = os.path.join(base_path,"dataset/chexpertchestxrays-u20210408")

#define the transforms associated to the CXP dataset
normalize_cxp = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

train_transform_cxp = transforms.Compose([transforms.RandomHorizontalFlip(),
                                          transforms.RandomRotation(15),
                                          transforms.Resize((256,256)),
                                          transforms.CenterCrop(256),
                                          transforms.ToTensor(),
                                          normalize_cxp])
val_transform_cxp = transforms.Compose([transforms.Resize(256),
                                        transforms.Resize((256,256)),
                                        transforms.ToTensor(),
                                        normalize_cxp])
test_transform_cxp = transforms.Compose([transforms.Resize((256,256)),
                                         transforms.CenterCrop(256),
                                         transforms.ToTensor(),
                                         normalize_cxp])

#tasks_labels: [3,17,18] from CXP, [3,17,18] from NIH, [0,7,9,12,16] from CXP, [1,2,4,5] from CXP, [1,2,4,5] from NIH
# [6,8,10,11] from NIH, [13,14,15] from NIH

#domain incremental, class incremental, class incremental, class incremental, domain incremental, 
#class incremental, class incremental
tasks_labels = [[3,17,18],[3,17,18],[0,7,9,12,16],[1,2,4,5],[1,2,4,5],[6,8,10,11],[13,14,15]]

#read the indices associated to each task from the corresponding txt file
#train_indices_tasks is a list of lists. The i-th list contains the indices of the images in the train_dataset where
#at least one of the pathologies of tasks_labels[i] appears.

file_path = os.path.join(base_path,"indices/NewScenario/train_indices_tasks.txt")

# Initialize train_indices_tasks as an empty list
train_indices_tasks = []

# Open the file for reading
with open(file_path, "r") as f:
    # Read each line of the file
    for line in f:
        # Split the line into individual indices (assuming indices are separated by spaces)
        indices = line.strip().split()
        # Convert indices from strings to integers and append them to train_indices_tasks
        train_indices_tasks.append([int(index) for index in indices])

file_path = os.path.join(base_path,"indices/NewScenario/val_indices_tasks.txt")

# Initialize train_indices_tasks as an empty list
val_indices_tasks = []

# Open the file for reading
with open(file_path, "r") as f:
    # Read each line of the file
    for line in f:
        # Split the line into individual indices (assuming indices are separated by spaces)
        indices = line.strip().split()
        # Convert indices from strings to integers and append them to train_indices_tasks
        val_indices_tasks.append([int(index) for index in indices])

file_path = os.path.join(base_path,"indices/NewScenario/test_indices_tasks.txt")

# Initialize train_indices_tasks as an empty list
test_indices_tasks = []

# Open the file for reading
with open(file_path, "r") as f:
    # Read each line of the file
    for line in f:
        # Split the line into individual indices (assuming indices are separated by spaces)
        indices = line.strip().split()
        # Convert indices from strings to integers and append them to train_indices_tasks
        test_indices_tasks.append([int(index) for index in indices])

#indices of the tasks in tasks_labels associated to CXP
tasks_labels_cxp = [0,2,3]
#indices of the tasks in tasks_labels associated to NIH
tasks_labels_nih = [1,4,5,6]

#starting from the dataframes and the transforms, and given the indices in train_indices_tasks, val_indices_tasks,
#test_indices_tasks, define the 7 training datasets, the 7 validation dataset and the 7 test_datasets
train_datasets, val_datasets, test_datasets = create_datasets(dfs_cxp, dfs_nih, train_indices_tasks, val_indices_tasks,
                                                              test_indices_tasks, tasks_labels_cxp, tasks_labels_nih, 
                                                              path_image_cxp, train_transform_cxp, val_transform_cxp,
                                                              test_transform_cxp, path_image_nih, train_transform_nih,
                                                              val_transform_nih, test_transform_nih)

#compute the dataloaders starting from the datasets
train_dataloaders, val_dataloaders, test_dataloaders = create_dataloaders(train_datasets, val_datasets, test_datasets,
                                                                         tasks_labels_cxp, tasks_labels_nih)

#compute the joint datasets, hence the union of train_datasets, the union of val_datasets and the union of 
#test_datasets, considering only the tasks associated to CXP
train_dataset_joint_cxp, val_dataset_joint_cxp, test_dataset_joint_cxp, path_list_train, path_list_val = create_datasets_joint_cxp(train_indices_tasks, 
                                                                        val_indices_tasks, test_indices_tasks, train_df_cxp, 
                                                                        val_df_cxp, test_df_cxp, path_image_cxp, train_transform_cxp,
                                                                        val_transform_cxp, test_transform_cxp, tasks_labels_cxp)

#compute the joint datasets, hence the union of train_datasets, the union of val_datasets and the union of 
#test_datasets, considering only the tasks associated to NIH
train_dataset_joint_nih, val_dataset_joint_nih, test_dataset_joint_nih = create_datasets_joint_nih(train_indices_tasks, 
                                                                        val_indices_tasks, test_indices_tasks, train_df_nih, 
                                                                        val_df_nih, test_df_nih, path_image_nih, train_transform_nih,
                                                                        val_transform_nih, test_transform_nih, tasks_labels_nih)

#Put the joint datasets in a list so that in the components of tasks_labels_cxp we have the cxp datasets (and viceversa)
# These datasets are used to test the model on previous tasks
val_datasets_joint = [val_dataset_joint_cxp,val_dataset_joint_nih,val_dataset_joint_cxp,val_dataset_joint_cxp,
                      val_dataset_joint_nih,val_dataset_joint_nih,val_dataset_joint_nih]

test_datasets_joint = [test_dataset_joint_cxp,test_dataset_joint_nih,test_dataset_joint_cxp,test_dataset_joint_cxp,
                      test_dataset_joint_nih,test_dataset_joint_nih,test_dataset_joint_nih]

#compute the associated dataloaders
val_dataloader_joint_cxp = DataLoader(val_dataset_joint_cxp, batch_size=48, shuffle=True, num_workers = 12, 
                                      pin_memory = True)
test_dataloader_joint_cxp = DataLoader(test_dataset_joint_cxp, batch_size=48, shuffle=True, num_workers = 12, 
                                      pin_memory = True)

val_dataloader_joint_nih = DataLoader(val_dataset_joint_nih, batch_size=32, shuffle=True, num_workers = 12, 
                                      pin_memory = True)
test_dataloader_joint_nih = DataLoader(test_dataset_joint_nih, batch_size=32, shuffle=True, num_workers = 12, 
                                      pin_memory = True)

#Put the joint datasets in a list so that in the components of tasks_labels_cxp we have the cxp datasets (and viceversa)
test_dataloaders_joint = [test_dataloader_joint_cxp,test_dataloader_joint_nih,test_dataloader_joint_cxp,
                          test_dataloader_joint_cxp,test_dataloader_joint_nih,test_dataloader_joint_nih,
                          test_dataloader_joint_nih]

val_dataloaders_joint = [val_dataloader_joint_cxp,val_dataloader_joint_nih,val_dataloader_joint_cxp,
                          val_dataloader_joint_cxp,val_dataloader_joint_nih,val_dataloader_joint_nih,
                          val_dataloader_joint_nih]

#Model: a 121-layer DenseNet with pre-trained weights from ImageNet
model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=True)

# we go from a classification of 1014 classes to classification of 19 classes
model.classifier = nn.Sequential(
    nn.Linear(1024, 19),
    #The output of the network is an array of 19 numbers between 0 and 1 indicating the probability of each disease label
    nn.Sigmoid()
)
#multi-label binary cross entropy loss
criterion = torch.nn.BCELoss()
#We use Adam optimization with default parameters,  initial LR of 0.0005
optimizer = optim.Adam(model.parameters(), lr=0.0005)

model.to(device)

#variable needed to modify the learning rate if the validation loss doesn't improve over three epochs
epochs_no_improve = 0

#initialize the best validation loss to the highest possible value
best_val_loss = float('inf')

for i in range(len(train_dataloaders)):
    #initialize the best validation loss to the highest possible value

    best_val_loss = float('inf')
    #variable needed to stop learning if the validation loss doesn't improve over ten epochs
    epochs_no_improve = 0

    num_epochs = 10
    
    # Training loop
    for epoch in range(num_epochs):
        # Set model to training mode
        model.train()
        
        #train the model for one epoch
        train_model_naive(train_dataloaders, device, model, optimizer, criterion, tasks_labels, epoch, i)
        
        model.eval()
        #compute the validation loss
        val_loss = eval_model_naive(val_dataloader_joint_cxp, val_dataloader_joint_nih, device, model, 
                                    tasks_labels, i, tasks_labels_cxp, tasks_labels_nih, criterion )

        #if the val loss is the best seen so far, update the corresponding parameters
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_epoch = epoch

        else:
            epochs_no_improve += 1

        # Adjust learning rate if the validation loss has not improved for three epochs
        if epochs_no_improve >= 3:
            old_lr = optimizer.param_groups[0]['lr']
            new_lr = old_lr / 2.0
            print(f"Adjusting learning rate from {old_lr} to {new_lr}")
            optimizer.param_groups[0]['lr'] = new_lr

    #re-initialize the parameters
    optimizer.param_groups[0]['lr'] = 0.0005 
    #save the model in memory
    torch.save(model.state_dict(), os.path.join(base_path,'/models/model_naive_task{0}_epoch{1}.pth'.format(i,10)))

model.eval()

best_thresholds = {}

num_classes = 19

#evaluate each model (trained on each task) on all tasks up to the tasks it was trained on
#compute AUC and F1 score
for p in range(len(test_datasets_joint)):
    print("\nTESTING MODEL TRAINED ON TASK", p)
    state_dict = torch.load(os.path.join(base_path,'/models/model_Naive_task{0}_epoch{1}.pth'.format(p,10)))
    model.load_state_dict(state_dict)
    
    compute_auc_and_f1(p, test_datasets_joint, test_dataloaders_joint, device, model, val_datasets_joint, 
                       val_dataloaders_joint, num_classes, tasks_labels, best_thresholds)
