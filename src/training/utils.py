from src.data.utils import filter_single_target
from src.data.utils import filter_target
from torch.utils.data import Subset, DataLoader
import torch
import time
import random
from sklearn.metrics import roc_auc_score
import os
import numpy as np

def train_model_joint(train_dataloader_joint, device, model, path_list_train, optimizer, criterion, reference_vectors, epoch):
    start_time = time.time()
    running_loss = 0.0
    for m,data in enumerate(train_dataloader_joint):
        #initialize lists that will contain the ground truth and output batches
        target_batches_cxp = []
        target_batches_nih = []
        output_batches_cxp = []
        output_batches_nih = []

        #extract the batch sample
        img_batch = data[0].to(device)
        target_batch = data[1].to(device)
        indices_batch = data[2]
        output_batch = model(img_batch).to(device)

        for i in range(len(target_batch)):
            #if the sample comes from cxp, filter it in order to consider only the pathologies of cxp
            if(indices_batch[i] in path_list_train):
                target = filter_single_target(target_batch[i], reference_vectors[0], device)
                output = filter_single_target(output_batch[i], reference_vectors[0], device)
                output_batches_cxp.append(output)
                target_batches_cxp.append(target)
            #if the sample comes from nih, filter it in order to consider only the pathologies of nih
            else:
                target = filter_single_target(target_batch[i], reference_vectors[1], device)
                output = filter_single_target(output_batch[i], reference_vectors[1], device)
                output_batches_nih.append(output)
                target_batches_nih.append(target)

        loss1 = torch.tensor(0.0)
        loss2 = torch.tensor(0.0)
        # Stack the target batches
        if(len(target_batches_cxp)>0):
            stacked_target_cxp = torch.stack(target_batches_cxp)
            stacked_output_cxp = torch.stack(output_batches_cxp)
            loss1 = criterion(stacked_output_cxp,stacked_target_cxp)
        if(len(target_batches_nih)>0):
            stacked_target_nih = torch.stack(target_batches_nih)
            stacked_output_nih = torch.stack(output_batches_nih)
            loss2 = criterion(stacked_output_nih,stacked_target_nih)

        #compute the loss as the average between the two losses
        if(loss1.item()!=0 and loss2.item()!=0):
            loss = (loss1*len(target_batches_cxp)+loss2*len(target_batches_nih))/(len(target_batches_cxp)+len(target_batches_nih))
        elif loss1.item() == 0:
            loss = loss2
        elif loss2.item() == 0:
            loss = loss1

        # Zero the parameter gradients
        optimizer.zero_grad()
        #backward
        loss.backward()
        #optimize
        optimizer.step()

        #update the loss
        running_loss += loss.item()
        #every 100 mini-batches, print the training loss and the time needed to process 100 mini-batches
        if m % 100 == 99:
            elapsed_time = time.time() - start_time
            print('[%d, %5d] loss: %.3f | time: %.2fs' % (epoch + 1, m + 1, running_loss / 100, elapsed_time))
            running_loss = 0.0
            start_time = time.time()    
            
def train_model_naive(train_dataloaders, device, model, optimizer, criterion, tasks_labels, epoch, i):
    #initialize the loss
    running_loss = 0.0
    start_time = time.time()
    
    task_labels = []
    for sublist in tasks_labels[:i+1]:
        task_labels.extend(set(sublist))

    task_labels = list(set(task_labels))

    #for each batch from the dataloader
    for m,data in enumerate(train_dataloaders[i]):
        #extract the batch sample
        img = data[0].to(device)
        target_batch = data[1].to(device)

        #modify the target_batch so that only the labels relative to the current task can be positive
        modified_target_batch = torch.zeros_like(target_batch)
        for k in tasks_labels[i]:
            modified_target_batch[:, k] = target_batch[:, k]

        #Forward
        output_batch = model(img).to(device)

        #filter the target to consider only the labels seen so far
        new_target = filter_target(modified_target_batch, task_labels, device)
        new_output = filter_target(output_batch, task_labels, device)
        #calculate the loss
        loss = criterion(new_output, new_target)
        # Zero the parameter gradients
        optimizer.zero_grad()
        #backward
        loss.backward()
        #optimize
        optimizer.step()

        #update the loss
        running_loss += loss.item()
        if m % 100 == 99:    # Print every 100 batches
            elapsed_time = time.time() - start_time
            print('[%d, %5d] loss: %.3f | time: %.2fs' % (epoch + 1, m + 1, running_loss / 100, elapsed_time))
            running_loss = 0.0
            start_time = time.time()
            
def train_model_lwf(train_dataloaders, device, model, optimizer, criterion, tasks_labels, epoch, i, old_targets):

    task_labels = []
    for sublist in tasks_labels[:i+1]:
        task_labels.extend(set(sublist))

    task_labels = list(set(task_labels))

    old_task_labels = []
    for sublist in tasks_labels[:i]:
        old_task_labels.extend(set(sublist))

    old_task_labels = list(set(old_task_labels))

    running_loss = 0.0
    start_time = time.time()

    #for each batch from the dataloader
    for m,data in enumerate(train_dataloaders[i]):
        #extract the batch sample
        img = data[0].to(device)
        target_batch = data[1].to(device)
        modified_target_batch = torch.zeros_like(target_batch).to(device)  # Ensure this tensor is on GPU
        for k in tasks_labels[i]:
            modified_target_batch[:, k] = target_batch[:, k]
        output_batch = model(img).to(device)
        idx_batch = data[2]

        filtered_target = filter_target(modified_target_batch, tasks_labels[i], device)
        filtered_output = filter_target(output_batch, tasks_labels[i], device)

        curr_loss = criterion(filtered_output, filtered_target)
        loss = curr_loss
        if i > 0:
            dist_loss = 0.0
            for j in range(len(idx_batch)):
                old_output = old_targets[idx_batch[j]]
                new_output = output_batch[j]

                filtered_old_output = filter_single_target(old_output, old_task_labels, device)
                filtered_new_output = filter_single_target(new_output, old_task_labels, device)
                dist_loss += criterion(filtered_new_output,filtered_old_output)
            dist_loss /= len(idx_batch)
            loss = curr_loss + 2*dist_loss

        # Zero the parameter gradients
        optimizer.zero_grad()
        #backward
        loss.backward()
        #optimize
        optimizer.step()

        #update the loss
        running_loss += loss.item()
        if m % 100 == 99:    # Print every 100 batches
            elapsed_time = time.time() - start_time
            print('[%d, %5d] loss: %.3f | time: %.2fs' % (epoch + 1, m + 1, running_loss / 100, elapsed_time))
            running_loss = 0.0
            start_time = time.time()

def compute_old_targets(old_targets, model, train_dataloaders, i, device):
    model.eval()
    with torch.no_grad():
        for m,data in enumerate(train_dataloaders[i]):
            img = data[0].to(device)
            idx_batch = data[2]
            output_batch = model(img).to(device)
            for j in range(len(idx_batch)):
                old_targets[idx_batch[j]] = output_batch[j]
                
                
def compute_old_targets_replay(old_targets, model, train_dataloaders, p, device, new_replayed_datasets):    
    model.eval()
    with torch.no_grad():
        for m,data in enumerate(train_dataloaders[p]):
            img = data[0].to(device)
            idx_batch = data[2]
            output_batch = model(img).to(device)
            for j in range(len(idx_batch)):
                old_targets[idx_batch[j]] = output_batch[j]
        replay_dataloader = DataLoader(new_replayed_datasets[p-1], batch_size = 48, num_workers = 12, pin_memory = True)
        for m,data in enumerate(replay_dataloader):
            img = data[0].to(device)
            idx_batch = data[2]
            output_batch = model(img).to(device)
            for j in range(len(idx_batch)):
                old_targets[idx_batch[j]] = output_batch[j]
                
                
def train_model_replay(train_dataloaders, device, model, optimizer, criterion, tasks_labels, epoch, p, tasks_labels_nih, tasks_labels_cxp, new_replayed_datasets):
    task_labels = []
    for sublist in tasks_labels[:p+1]:
        task_labels.extend(set(sublist))

    task_labels = list(set(task_labels))
    
    running_loss = 0.0
    start_time = time.time()

    # Set model to training mode

    #for each batch from the dataloader
    for m,data in enumerate(train_dataloaders[p]):
        #extract the batch sample
        img = data[0].to(device)
        target_batch = data[1].to(device)
        modified_target_batch = torch.zeros_like(target_batch).to(device)  # Ensure this tensor is on GPU
        for k in tasks_labels[i]:
            modified_target_batch[:, k] = target_batch[:, k]

        if p > 0:
            batch_size = 48
            if p in tasks_labels_nih:
                batch_size = 32
            replayed_dataset = new_replayed_datasets[p - 1]
            replayed_indices = random.sample(range(len(replayed_dataset)), batch_size)
            replayed_targets = [replayed_dataset[idx][1] for idx in replayed_indices]
            replayed_imgs = [replayed_dataset[idx][0] for idx in replayed_indices]

            replayed_target_batch = torch.stack(replayed_targets).to(device)
            replayed_img_batch = torch.stack(replayed_imgs).to(device)

            new_img = torch.cat([img, replayed_img_batch], dim=0).to(device)
            new_target_batch = torch.cat([modified_target_batch, replayed_target_batch], dim=0).to(device)
        else:
            new_img = img
            new_target_batch = modified_target_batch

        #Forward
        output_batch = model(new_img).to(device)
        new_target = filter_target(new_target_batch, task_labels, device)
        new_output = filter_target(output_batch, task_labels, device)
        #calculate the loss
        loss = criterion(new_output, new_target)
        # Zero the parameter gradients
        optimizer.zero_grad()
        #backward
        loss.backward()
        #optimize
        optimizer.step()

        #update the loss
        running_loss += loss.item()
        if m % 100 == 99:    # Print every 100 batches
            elapsed_time = time.time() - start_time
            print('[%d, %5d] loss: %.3f | time: %.2fs' % (epoch + 1, m + 1, running_loss / 100, elapsed_time))
            running_loss = 0.0
            start_time = time.time()
            
def train_model_pseudolabel(train_dataloaders, device, new_model, tasks_labels, i, model, criterion, optimizer, epoch, best_thresholds):
    task_labels = []
    for sublist in tasks_labels[:i+1]:
        task_labels.extend(set(sublist))

    task_labels = list(set(task_labels))

    old_task_labels = []
    for sublist in tasks_labels[:i]:
        old_task_labels.extend(set(sublist))

    old_task_labels = list(set(old_task_labels))
    
    #initialize the loss
    running_loss = 0.0
    start_time = time.time()

    #for each batch from the dataloader
    for m,data in enumerate(train_dataloaders[i]):
        #extract the batch sample
        img = data[0].to(device)
        target_batch = data[1].to(device)
        new_output = new_model(img).to(device)

        modified_target_batch = torch.zeros_like(target_batch)
        for k in tasks_labels[i]:
            modified_target_batch[:, k] = target_batch[:, k]

        if i > 0:
            for l in range(modified_target_batch.size(0)):
                for j in old_task_labels:
                    if new_output[l][j] > best_thresholds[j]:
                        modified_target_batch[l][j] = 1

        #Forward
        output_batch = model(img).to(device)

        new_target = filter_target(modified_target_batch, task_labels, device)
        new_output = filter_target(output_batch, task_labels, device)
        #calculate the loss
        loss = criterion(new_output, new_target)
        # Zero the parameter gradients
        optimizer.zero_grad()
        #backward
        loss.backward()
        #optimize
        optimizer.step()

        #update the loss
        running_loss += loss.item()
        if m % 100 == 99:    # Print every 100 batches
            elapsed_time = time.time() - start_time
            print('[%d, %5d] loss: %.3f | time: %.2fs' % (epoch + 1, m + 1, running_loss / 20, elapsed_time))
            running_loss = 0.0
            start_time = time.time()
            
def train_model_lwf_replay(train_dataloaders, tasks_labels, p, tasks_labels_nih, new_replayed_datasets, model, criterion, device, optimizer, epoch, old_targets):
    task_labels = []
    for sublist in tasks_labels[:p+1]:
        task_labels.extend(set(sublist))

    task_labels = list(set(task_labels))

    old_task_labels = []
    for sublist in tasks_labels[:p]:
        old_task_labels.extend(set(sublist))

    old_task_labels = list(set(old_task_labels))
    
    #initialize the loss
    running_loss = 0.0
    start_time = time.time()

    #for each batch from the dataloader
    for m,data in enumerate(train_dataloaders[p]):
        #extract the batch sample
        img = data[0].to(device)
        target_batch = data[1].to(device)
        modified_target_batch = torch.zeros_like(target_batch).to(device)  # Ensure this tensor is on GPU
        for k in tasks_labels[i]:
            modified_target_batch[:, k] = target_batch[:, k]

        if p > 0:
            batch_size = 48
            if p in tasks_labels_nih:
                batch_size = 32
            replayed_dataset = new_replayed_datasets[p - 1]
            replayed_indices = random.sample(range(len(replayed_dataset)), batch_size)
            replayed_targets = [replayed_dataset[idx][1] for idx in replayed_indices]
            replayed_imgs = [replayed_dataset[idx][0] for idx in replayed_indices]

            replayed_target_batch = torch.stack(replayed_targets).to(device)
            replayed_img_batch = torch.stack(replayed_imgs).to(device)

            new_img = torch.cat([img, replayed_img_batch], dim=0).to(device)
            new_target_batch = torch.cat([modified_target_batch, replayed_target_batch], dim=0).to(device)
        else:
            new_img = img
            new_target_batch = modified_target_batch

        new_output_batch = model(new_img).to(device)
        idx_batch = data[2]

        filtered_target = filter_target(new_target_batch, task_labels, device)
        filtered_output = filter_target(new_output_batch, task_labels, device)

        curr_loss = criterion(filtered_output, filtered_target)
        loss = curr_loss
        if p > 0:
            dist_loss = 0.0
            for j in range(len(idx_batch)):
                old_output = old_targets[idx_batch[j]]
                new_output = new_output_batch[j]

                filtered_old_output = filter_single_target(old_output, old_task_labels, device)
                filtered_new_output = filter_single_target(new_output, old_task_labels, device)
                dist_loss += criterion(filtered_new_output,filtered_old_output)
            dist_loss /= len(idx_batch)
            loss = curr_loss + 2*dist_loss

        # Zero the parameter gradients
        optimizer.zero_grad()
        #backward
        loss.backward()
        #optimize
        optimizer.step()

        #update the loss
        running_loss += loss.item()
        if m % 100 == 99:    # Print every 100 batches
            elapsed_time = time.time() - start_time
            print('[%d, %5d] loss: %.3f | time: %.2fs' % (epoch + 1, m + 1, running_loss / 20, elapsed_time))
            running_loss = 0.0
            start_time = time.time()
            
def train_model_rclp(train_dataloaders, tasks_labels, i, model, old_model, device, criterion, optimizer, epoch, replayed_datasets, tasks_labels_nih, best_thresholds, val_dataloaders,
                    subset_size, train_datasets, val_datasets, val_dataloaders, tasks_labels, base_path,  num_classes):
       task_labels = []
        for sublist in tasks_labels[:i+1]:
            task_labels.extend(set(sublist))

        task_labels = list(set(task_labels))

        old_task_labels = []
        for sublist in tasks_labels[:i]:
            old_task_labels.extend(set(sublist))

        old_task_labels = list(set(old_task_labels))

        # Training loop
        for epoch in range(num_epochs):

            model.train()
            #initialize the loss
            running_loss = 0.0
            start_time = time.time()

            for m,data in enumerate(train_dataloaders[i]):
                #extract the batch sample
                img = data[0].to(device)
                target_batch = data[1].to(device)

                modified_target_batch = torch.zeros_like(target_batch).to(device)  # Ensure this tensor is on GPU
                for k in tasks_labels[i]:
                    modified_target_batch[:, k] = target_batch[:, k]

                old_output = old_model(img).to(device)

                target_batch_clone = modified_target_batch.clone()

                if i > 0:
                    block12_extractor_model = DenseNet121Block12Extractor(model).to(device)
                    block12_extractor_old_model = DenseNet121Block12Extractor(old_model).to(device)
                    block12_extractor_model.eval()  # Set to evaluation mode
                    block12_extractor_old_model.eval()

                    for l in range(target_batch_clone.size(0)):
                        for j in old_task_labels:
                            if old_output[l][j] > best_thresholds[j]:
                                target_batch_clone[l][j] = 1
                                # Initialize replayed_dataset as None outside the loop

                    replayed_dataset = replayed_datasets[i][0]

                    if i > 1:
                        for k in range(1,len(replayed_datasets[i])):
                            replayed_dataset = MergedDataset(replayed_dataset, replayed_datasets[i][k])

                    batch_size = 48
                    if i in tasks_labels_nih:
                        batch_size = 32
                    replayed_indices = random.sample(range(len(replayed_dataset)), batch_size)

                    replayed_targets = [replayed_dataset[idx][1] for idx in replayed_indices]
                    replayed_target_batch = torch.stack(replayed_targets).to(device)

                    replayed_img_batch = torch.stack([replayed_dataset[idx][0] for idx in replayed_indices]).to(device)
                    replayed_output = model(replayed_img_batch).to(device)

                    filtered_target_batch = filter_target(replayed_target_batch, old_task_labels, device)
                    filtered_output = filter_target(replayed_output, old_task_labels, device)

                    loss2 = criterion(filtered_output, filtered_target_batch)

                    features_model = extract_features(block12_extractor_model, img).to(device)
                    features_old_model = extract_features(block12_extractor_old_model, img).to(device)

                    replayed_features_model = extract_features(block12_extractor_model, replayed_img_batch).to(device)
                    replayed_features_old_model = extract_features(block12_extractor_old_model, replayed_img_batch).to(device)

                    dist_loss1 = torch.nn.MSELoss()(features_model, features_old_model)
                    dist_loss2 = torch.nn.MSELoss()(replayed_features_model, replayed_features_old_model)
                    dist_loss = (dist_loss1 + dist_loss2)/2

                #Forward
                output_batch = model(img).to(device)
                new_target = filter_target(target_batch_clone, task_labels, device)
                new_output = filter_target(output_batch, task_labels, device)
                #calculate the loss
                loss1 = criterion(new_output, new_target)
                if i == 0:
                    loss = loss1
                else:
                    loss = (loss1+loss2)/2 + dist_loss
                # Zero the parameter gradients
                optimizer.zero_grad()
                #backward
                loss.backward()
                #optimize
                optimizer.step()

                #update the loss
                running_loss += loss.item()
                if m % 100 == 99:    # Print every 100 batches
                    elapsed_time = time.time() - start_time
                    print('[%d, %5d] loss: %.3f | time: %.2fs' % (epoch + 1, m + 1, running_loss / 100, elapsed_time))
                    running_loss = 0.0
                    start_time = time.time()

            model.eval()
            #compute and print the validation loss
            with torch.no_grad():
                val_loss = 0.0

                for k, val_data in enumerate(val_dataloaders[i]):
                    #extract the current batch
                    val_img = val_data[0].to(device)
                    val_target_batch = val_data[1].to(device)
                    val_output_batch = model(val_img).to(device)

                    new_target = filter_target(val_target_batch, tasks_labels[i], device).to(device)
                    new_output = filter_target(val_output_batch, tasks_labels[i], device).to(device)  # Ensure tensors are on GPU
                    #calculate the loss
                    loss = criterion(new_output, new_target).to(device)
                    val_loss += loss.item()
                val_loss /= (k + 1)
                print("Validation loss: ", val_loss) #average on the number of batches

        torch.save(model.state_dict(), state_dict = torch.load(os.path.join(base_path,'/models/RCLP_gamma1_block12_task{0}_epoch'.format(i,10)))

        j = i+1
        subset_datasets = []
        new_subset_size = int(subset_size/(j))

        #sample the subset from the original dataset
        indices = list(range(len(train_datasets[i])))
        random_indices = random.sample(indices, new_subset_size)
        random_subset_dataset = Subset(train_datasets[i], random_indices)
        subset_datasets = []

        if i > 0:
            old_model.eval()
            with torch.no_grad():
                modified_dataset = []
                dataloader = DataLoader(random_subset_dataset, batch_size=32, shuffle=False)
                for batch in dataloader:
                    img_batch = batch[0].to(device)
                    target_batch = batch[1].to(device)

                    modified_target_batch = torch.zeros_like(target_batch).to(device)  # Ensure this tensor is on GPU
                    for k in tasks_labels[i]:
                        modified_target_batch[:, k] = target_batch[:, k]

                    idx = batch[2]
                    old_model_output = old_model(img_batch).to(device)

                    for l in range(target_batch.size(0)):
                        for n in old_task_labels:
                            if old_model_output[l][n] > best_thresholds[n]:
                                modified_target_batch[l][n] = 1
                        modified_dataset.append((img_batch[l], modified_target_batch[l], idx[l]))
            subset_datasets.append(modified_dataset)
        else:
            with torch.no_grad():
                modified_dataset = []
                dataloader = DataLoader(random_subset_dataset, batch_size=32, shuffle=False)
                for batch in dataloader:
                    img_batch = batch[0].to(device)
                    target_batch = batch[1].to(device)

                    modified_target_batch = torch.zeros_like(target_batch).to(device)  # Ensure this tensor is on GPU
                    for k in tasks_labels[i]:
                        modified_target_batch[:, k] = target_batch[:, k]

                    idx = batch[2]

                    for l in range(target_batch.size(0)):
                        modified_dataset.append((img_batch[l], modified_target_batch[l], idx[l]))

            subset_datasets.append(modified_dataset)

        with torch.no_grad():
            num_val_samples = len(val_datasets_joint[i])
            #initialize the two matrices with all zero elements
            val_outputs = np.zeros((num_classes, num_val_samples))
            val_targets = np.zeros((num_classes, num_val_samples))

            for j,val_data in enumerate(val_dataloaders_joint[i]):
                #extract the current batch
                val_img = val_data[0].to(device)
                val_target = val_data[1].to(device)

                #Forward pass
                val_output = model(val_img).to(device)

                batch_size = val_target.size(0)
                #index of the first sample in the batch (the first batch is saved in the columns from 0 to 31, the second from 32 to 63...)
                index_first_sample = j*batch_size
                #iterate from index_first_sample to index_first_sample + batch_size - 1
                for k in range(0, batch_size):
                    for l in range(num_classes):
                        val_outputs[l, index_first_sample + k] = val_output[k][l].item()
                        val_targets[l, index_first_sample + k] = val_target[k][l].item()

                num_classes = len(val_outputs)
                thresholds = np.arange(0,1.0001,0.0001).astype(np.float32)
                best_f1_scores = {}
                overall_f1_score = 0

            #for every pathology, and for every threshold, we compute the f1 score relative to that pathology using that threshold, saving the best pathology and the best threshold
            for pathology in tasks_labels[i]:
                #consider the lines of thee two matrices relative to the current pathology
                pathology_targets = val_targets[pathology, :]
                pathology_outputs = val_outputs[pathology, :]

                num_thresholds = len(thresholds)
                #initialize the vector of f1 scores, one for each threshold
                f1_scores = np.zeros(num_thresholds)

                for l,threshold in enumerate(thresholds):
                    #convert the vector of pathology_ouputs in binary form using the current threshold
                    binary_outputs = pathology_outputs > threshold

                    #compute the true positics, false positives and false negatives
                    true_positives = np.sum((binary_outputs == 1) & (pathology_targets == 1))
                    false_positives = np.sum((binary_outputs == 1) & (pathology_targets == 0))
                    false_negatives = np.sum((binary_outputs == 0) & (pathology_targets == 1))

                    #compute the f1 score
                    precision = true_positives / (true_positives + false_positives + 1e-7)
                    recall = true_positives / (true_positives + false_negatives + 1e-7)
                    f1_score = 2 * precision * recall / (precision + recall + 1e-7)

                    #add the value to the vector
                    f1_scores[l] = f1_score

                #extract the maximum of the vector with all the f1 scores relative to all thresholds
                best_index = np.argmax(f1_scores)

                #find the relative best threshold and save it in the dictionary
                if pathology not in best_thresholds.keys():
                    best_thresholds[pathology] = thresholds[best_index]

            print("Best F1 score thresholds: ", best_thresholds)

        #save the model in memory
        state_dict = torch.load(os.path.join(base_path,'/models/RCLP_gamma1_block12_task{0}_epoch'.format(i,10)))
        #update the teacher
        old_model.load_state_dict(state_dict)
        old_model.eval()

        if i > 0:
            for k in range(len(replayed_datasets[i])):       
                #sample the subset from the original dataset
                indices = list(range(len(replayed_datasets[i][k])))
                random_indices = random.sample(indices, new_subset_size)
                random_subset_dataset = Subset(replayed_datasets[i][k], random_indices)

                modified_dataset = []
                dataloader = DataLoader(random_subset_dataset, batch_size=32, shuffle=False)
                for batch in dataloader:
                    img_batch = batch[0].to(device)
                    target_batch = batch[1].to(device)
                    idx = batch[2]
                    old_model_output = old_model(img_batch).to(device)

                    for l in range(target_batch.size(0)):
                        for n in tasks_labels[i]:
                            if old_model_output[l][n] > best_thresholds[n]:
                                target_batch[l][n] = 1
                        modified_dataset.append((img_batch[l], target_batch[l], idx[l]))
                #append the subset to the buffer
                subset_datasets.append(modified_dataset)

            #append the buffer to the list of buffers
        replayed_datasets.append(subset_datasets)                  
