

# Pseudo-Label Replay for the NIC scenario in the Medical Domain
<div align="center">
<img src="imgs/ScenarioCL_Medical_2.png" width="700">
<div align="left">
  
* **NIC Scenario**: Implementation of the New Instances and New Classes (NIC) scenario for the medical domain.
* **Pseudo-Label Replay**: Implementation of a new Continual Learning (CL) method that minimizes forgetting.
****

# Introduction
<div align="center">
<img src="imgs/taskDiseasePrevalence.png" width="700">
<div align="left">
We implement the NIC scenario in the medical domain considering the problem of pathology classification of Chest X-ray images.
In particular, we consider a scenario of 7 tasks, for a total of 19 pathologies, such that between two successive tasks either a domain shift occurs or new classes are introduced. The scenario is implemented using the CheXpert (CXP) and ChestXray14 (NIH) datasets.

To each task are associated all and only the samples in which at least one of the relative pathologies appears. 

The plots represented above represent the prevalence of each disease in each task. The red line is associated to the prevalence of the pathologies in the original dataset. The blue bars represent the occurrence of the pathologies associated to the relative task, while the gray lines represent the prevalence of the other - hidden - pathologies. As highlighted by the figure, there is some intersection between tasks, hence some samples are in common.

Moreover, we propose and implement a new method to overcome the limitations of traditional CL strategies in this scenario:
- Replay-based methods perform poorly due to the interference between replayed samples and new samples.
- Distillation-based methods require the re-appearance of old labels in new images.

<div align="center">
<img src="imgs/Data_v3.png" width="700">
<div align="left">
Our approach, called Pseudo-Label Replay, ties together the Pseudo-Label and Replay methods, in a smart way such that the knowledge acquired by the model is gradually added to the samples saved in the memory buffer.

For more details, please refer to our [paper](#citation)

## Results
<div align="center">
<img src="imgs/F1.png" width="400">
<div align="left">

We benchmarked the traditional CL methods and our novel method Pseudo-Label Replay on a NIC scenario in the medical domain.
We found that our approach outperforms existing methods in terms of forgetting and final F1 score.

## Training execution
To run our code, the download of the NIH and CXP datasets is necessary. The datasets can be downloaded respectively from https://stanfordmlgroup.github.io/competitions/chexpert/ and https://paperswithcode.com/dataset/chestx-ray14.

In the folder datasets_indices, one can find the .txt files from which the indices of the images belonging to the train, validation and test set are read by the scripts, both for the CXP and NIH dataset.

Moreover, the files train_indices_tasks.txt, val_indices_tasks.txt, test_indices_tasks.txt in this folder contain the indices of the images belonging to each task. The code to execute the division in tasks in available in the script create_scenario.py in the "src" folder. If one wants to modify the task stream, it's sufficient to modify this code. 

The folder "models" is empty; however, when running the script of each method, the models resulting from training on each task are saved in this folder in the form 'models/model_{method}_{taskID}_{epoch}.pth'.

In the src folder, a script for each method is present.

To execute each script it's sufficient to run "python {method}.py", where {method} needs to be replaced with the name of the strategy that one wants to use.

The scripts execute the training of the model using the corresponding strategy. During the training, the training loss is printed every 100 mini-batches and the validation loss is printed after each epoch. The evaluation of the resulting models is then computed on all the test sets relative to the old tasks and the current one, and the corresponding values of AUC and F1 score are printed.

The folders data, training, and eval contain scripts in which auxiliary functions, needed for training using each strategy, are defined.

****

## Citation

If you find this project useful in your research, please add a star and cite us 😊 

```BibTeX
@misc{Multi-Label Continual Learning for the Medical Domain: A Novel Benchmark,
    title={},
    author={},
    year={2024}
}
```

