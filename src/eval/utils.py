import torch
from src.data.utils import filter_single_target
from src.data.utils import filter_target
from sklearn.metrics import roc_auc_score
import numpy as np

def eval_model_joint(val_dataloader_joint, device, model, reference_vectors, path_list_val, criterion):
    val_loss = 0.0

    for k,val_data in enumerate(val_dataloader_joint):
        #initialize lists that will contain the ground truth and output batches
        target_batches_cxp = []
        target_batches_nih = []
        output_batches_cxp = []
        output_batches_nih = []
        #extract the batch sample
        img_batch = val_data[0].to(device)
        target_batch = val_data[1].to(device)
        indices_batch = val_data[2]
        output_batch = model(img_batch).to(device)

        for i in range(len(target_batch)):
            #if the sample comes from cxp, filter it in order to consider only the pathologies of cxp
            if(indices_batch[i] in path_list_val):
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
            loss = (loss1.item()*len(target_batches_cxp)+loss2.item()*len(target_batches_nih))/(len(target_batches_cxp)+len(target_batches_nih))
        elif loss1.item() == 0:
            loss = loss2.item()
        elif loss2.item() == 0:
            loss = loss1.item()
        #update the overall loss
        val_loss += loss

    val_loss /= (k+1)
    print("Validation loss: ", val_loss) #average on the number of batches
    
    return val_loss


def eval_model_naive(val_dataloader_joint_cxp, val_dataloader_joint_nih, device, model, tasks_labels, i, tasks_labels_cxp, tasks_labels_nih, criterion ):
    task_labels = []
    for sublist in tasks_labels[:i+1]:
        task_labels.extend(set(sublist))

    task_labels = list(set(task_labels))
    
    val_loss = 0.0

    if i in tasks_labels_cxp:
        val_dataloader = val_dataloader_joint_cxp
    elif i in tasks_labels_nih:
        val_dataloader = val_dataloader_joint_nih
    for k,val_data in enumerate(val_dataloader):
        #extract the current batch
        val_img = val_data[0].to(device)
        val_target_batch = val_data[1].to(device)

        modified_target_batch = torch.zeros_like(val_target_batch)
        for l in tasks_labels[i]:
            modified_target_batch[:, l] = val_target_batch[:, l]
        #Forward pass
        val_output_batch = model(val_img)

        new_target_val = filter_target(modified_target_batch, task_labels, device)
        new_output_val = filter_target(val_output_batch, task_labels, device)
        #calculate the loss
        loss = criterion(new_output_val, new_target_val)
        #update the overall loss
        val_loss += loss.item()

    val_loss /= (k+1)
    print("Validation loss: ", val_loss) #average on the number of batches
    
    return val_loss

def eval_model(val_dataloaders, i, device, model, criterion, tasks_labels):
    with torch.no_grad():
        val_loss = 0.0

        for k,val_data in enumerate(val_dataloaders[i]):
            #extract the current batch
            val_img = val_data[0].to(device)
            val_target_batch = val_data[1].to(device)
            val_output_batch = model(val_img)

            new_target = filter_target(val_target_batch, tasks_labels[i], device)
            new_output = filter_target(val_output_batch, tasks_labels[i], device)
            #calculate the loss
            loss = criterion(new_output, new_target)
            val_loss += loss.item()

        val_loss /= (k+1)
    print("Validation loss: ", val_loss) #average on the number of batches
    return val_loss

def compute_auc_and_f1(p, test_datasets_joint, test_dataloaders_joint, device, model, val_datasets_joint, val_dataloaders_joint, num_classes, tasks_labels, best_thresholds):
    with torch.no_grad():
        for m in range(p+1): 
            print("\nTESTING ON TASK", m)
            num_test_samples = len(test_datasets_joint[m])
            task_labels = tasks_labels[m]
            #initialize the two matrices with all zero elements
            test_outputs = np.zeros((num_classes, num_test_samples))
            test_targets = np.zeros((num_classes, num_test_samples))

            for j,test_data in enumerate(test_dataloaders_joint[m]):
                #extract the current batch
                test_img = test_data[0].to(device)
                test_target = test_data[1].to(device)

                #Forward pass
                test_output = model(test_img)

                batch_size = test_target.size(0)
                #index of the first sample in the batch (the first batch is saved in the columns from 0 to 31, the second from 32 to 63...)
                index_first_sample = j*batch_size
                #iterate from index_first_sample to index_first_sample + batch_size - 1
                for k in range(0, batch_size):
                    for l in range(num_classes):
                        test_outputs[l, index_first_sample + k] = test_output[k][l].item()
                        test_targets[l, index_first_sample + k] = test_target[k][l].item()


            num_val_samples = len(val_datasets_joint[m])
            #initialize the two matrices with all zero elements
            val_outputs = np.zeros((num_classes, num_val_samples))
            val_targets = np.zeros((num_classes, num_val_samples))

            for j,val_data in enumerate(val_dataloaders_joint[m]):
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

            print("\n*** AUC SCORES ON TEST SET ***")
            #we initialize the vector of auc scores to have 14 components equal to 0(14 = number of pathologies), in the following only the components corresponding to the pathologies of the relative task will be modified
            auc_scores = np.zeros(num_classes)

            for pathology in task_labels:
                #consider the whole lines of test_targets and test_outputs relative to the current pathology
                pathology_targets = test_targets[pathology, :]
                pathology_outputs = test_outputs[pathology, :]       

                #add it to the vector
                auc_score = roc_auc_score(pathology_targets, pathology_outputs)
                auc_scores[pathology] = auc_score

            for pathology in task_labels:
                print('Pathology %d AUC: %.3f' % (pathology, auc_scores[pathology]))

            # Print average AUC for all pathologies of the task
            avg_auc = np.sum(auc_scores)/len(task_labels)
            print("Average AUC: ", avg_auc)


            num_classes = len(val_outputs)
            thresholds = np.arange(0,1.0001,0.0001).astype(np.float32)
            best_f1_scores = {}
            overall_f1_score = 0

            if m==p:
                #for every pathology, and for every threshold, we compute the f1 score relative to that pathology using that threshold, saving the best pathology and the best threshold
                for pathology in task_labels:
                    #consider the lines of thee two matrices relative to the current pathology
                    pathology_targets = val_targets[pathology, :]
                    pathology_outputs = val_outputs[pathology, :]

                    num_thresholds = len(thresholds)
                    #initialize the vector of f1 scores, one for each threshold
                    f1_scores = np.zeros(num_thresholds)

                    for i,threshold in enumerate(thresholds):
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
                        f1_scores[i] = f1_score

                    #extract the maximum of the vector with all the f1 scores relative to all thresholds
                    best_index = np.argmax(f1_scores)

                    #find the relative best threshold and save it in the dictionary
                    if pathology not in best_thresholds.keys():
                        best_thresholds[pathology] = thresholds[best_index]

            print("Best F1 score thresholds: ", best_thresholds)

            print("\n**COMPUTATION OF THE F1 SCORES ON TEST DATASET**")
            overall_f1_score = 0

            #for every pathology, and for every threshold, we compute the f1 score relative to that pathology using that threshold, saving the best pathology and the best threshold
            for pathology in task_labels:
                #consider the lines of thee two matrices relative to the current pathology
                pathology_targets = test_targets[pathology, :]
                pathology_outputs = test_outputs[pathology, :]

                #convert the vector of pathology_ouputs in binary form using the current threshold
                binary_outputs = pathology_outputs > best_thresholds[pathology]

                #compute the true positics, false positives and false negatives
                true_positives = np.sum((binary_outputs == 1) & (pathology_targets == 1))
                false_positives = np.sum((binary_outputs == 1) & (pathology_targets == 0))
                false_negatives = np.sum((binary_outputs == 0) & (pathology_targets == 1))

                #compute the f1 score
                precision = true_positives / (true_positives + false_positives + 1e-7)
                recall = true_positives / (true_positives + false_negatives + 1e-7)
                f1_score = 2 * precision * recall / (precision + recall + 1e-7)

                print('Pathology %d F1: %.3f' % (pathology, f1_score))

                overall_f1_score += f1_score

            #compute the average f1 score
            avg_f1_score = overall_f1_score/len(task_labels)

            print('Overall F1 score: ', avg_f1_score.item())
            
            
def compute_auc_and_f1_joint(test_datasets_joint, test_dataloaders_joint, device, model, val_datasets_joint, val_dataloaders_joint, num_classes, tasks_labels):
    for m in range(len(test_datasets_joint)):
        model.eval()
        num_test_samples = len(test_datasets_joint[m])
        #initialize the two matrices with all zero elements
        test_outputs = np.zeros((num_classes, num_test_samples))
        test_targets = np.zeros((num_classes, num_test_samples))
        task_labels = tasks_labels[m]

        for j,test_data in enumerate(test_dataloaders_joint[m]):
            #extract the current batch
            test_img = test_data[0].to(device)
            test_target = test_data[1].to(device)

            #Forward pass
            test_output = model(test_img)

            batch_size = test_target.size(0)
            #index of the first sample in the batch (the first batch is saved in the columns from 0 to 31, the second from 32 to 63...)
            index_first_sample = j*batch_size
            #iterate from index_first_sample to index_first_sample + batch_size - 1
            for k in range(0, batch_size):
                for l in range(num_classes):
                    test_outputs[l, index_first_sample + k] = test_output[k][l].item()
                    test_targets[l, index_first_sample + k] = test_target[k][l].item()


        num_val_samples = len(val_datasets_joint[m])
        #initialize the two matrices with all zero elements
        val_outputs = np.zeros((num_classes, num_val_samples))
        val_targets = np.zeros((num_classes, num_val_samples))

        for j,val_data in enumerate(val_dataloaders_joint[m]):
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

        print("\n*** AUC SCORES ON TEST SET ***")
        #we initialize the vector of auc scores to have 14 components equal to 0(14 = number of pathologies), in the following only the components corresponding to the pathologies of the relative task will be modified
        auc_scores = np.zeros(num_classes)

        for pathology in task_labels:
            #consider the whole lines of test_targets and test_outputs relative to the current pathology
            pathology_targets = test_targets[pathology, :]
            pathology_outputs = test_outputs[pathology, :]       

            #add it to the vector
            auc_score = roc_auc_score(pathology_targets, pathology_outputs)
            auc_scores[pathology] = auc_score

        for pathology in task_labels:
            print('Pathology %d AUC: %.3f' % (pathology, auc_scores[pathology]))

        # Print average AUC for all pathologies of the task
        avg_auc = np.sum(auc_scores)/len(task_labels)
        print("Average AUC: ", avg_auc)


        num_classes = len(val_outputs)
        thresholds = np.arange(0,1.0001,0.0001).astype(np.float32)
        best_thresholds = {}
        best_f1_scores = {}
        overall_f1_score = 0

        #for every pathology, and for every threshold, we compute the f1 score relative to that pathology using that threshold, saving the best pathology and the best threshold
        for pathology in task_labels:
            #consider the lines of thee two matrices relative to the current pathology
            pathology_targets = val_targets[pathology, :]
            pathology_outputs = val_outputs[pathology, :]

            num_thresholds = len(thresholds)
            #initialize the vector of f1 scores, one for each threshold
            f1_scores = np.zeros(num_thresholds)

            for i,threshold in enumerate(thresholds):
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
                f1_scores[i] = f1_score

            #extract the maximum of the vector with all the f1 scores relative to all thresholds
            best_index = np.argmax(f1_scores)

            #find the relative best threshold and save it in the dictionary
            best_thresholds[pathology] = thresholds[best_index]

        print("Best F1 score thresholds: ", best_thresholds)

        print("\n**COMPUTATION OF THE F1 SCORES ON TEST DATASET**")
        overall_f1_score = 0

        #for every pathology, and for every threshold, we compute the f1 score relative to that pathology using that threshold, saving the best pathology and the best threshold
        for pathology in task_labels:
            #consider the lines of thee two matrices relative to the current pathology
            pathology_targets = test_targets[pathology, :]
            pathology_outputs = test_outputs[pathology, :]

            #convert the vector of pathology_ouputs in binary form using the current threshold
            binary_outputs = pathology_outputs > best_thresholds[pathology]

            #compute the true positics, false positives and false negatives
            true_positives = np.sum((binary_outputs == 1) & (pathology_targets == 1))
            false_positives = np.sum((binary_outputs == 1) & (pathology_targets == 0))
            false_negatives = np.sum((binary_outputs == 0) & (pathology_targets == 1))

            #compute the f1 score
            precision = true_positives / (true_positives + false_positives + 1e-7)
            recall = true_positives / (true_positives + false_negatives + 1e-7)
            f1_score = 2 * precision * recall / (precision + recall + 1e-7)

            print('Pathology %d F1: %.3f' % (pathology, f1_score))

            overall_f1_score += f1_score

        #compute the average f1 score
        avg_f1_score = overall_f1_score/len(task_labels)

        print('Overall F1 score: ', avg_f1_score.item())
