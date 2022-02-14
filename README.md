# Deep Clustering and Visualization (DCV)


This is a PyTorch implementation of the DCV, and the code includes the following modules:

* Datasets (MNIST, HAR, USPS, Pendigits, Reuters-10K, Coil100)

* Training for DCV-encoder and DCV-decoder

* Visualization

* Evaluation metrics 

  

## Requirements

* pytorch == 1.6.0

* scipy == 1.3.1

* numpy == 1.18.5

* scikit-learn == 0.21.3

* umap == 1.18.5

* networkx == 2.3

  

## Description

* main.py  
  * Train() -- Train a new model
  * Test() -- Test the learned model for evaluating generalization
* dataloader.py  
  
  * GetData() -- Load data of selected dataset
* model.py  
  
  * LISV2_MLP() -- model and loss
* tool.py  
  * GIFPloter() -- Auxiliary tool for online plot
  
  * DataSaver() -- Save intermediate and final results
  
  * cluster_acc() -- Calculate clustering accuracy
  



## Dataset

The datasets and pretrained models used in this paper are available in:

https://drive.google.com/file/d/19oO9l9WgnPZuqojKFVtwIRFm4s0vcY02/view?usp=sharing



## Running the code

1. Install the required dependency packages
2. To get the results on a specific *dataset*, run with proper hyperparameters

  ```
python main.py --data_name dataset
  ```

3. To get the data, metrics, and visualisation, refer to

  ```
../log/dataset/
  ```

where the *dataset* is one of the six datasets (MNIST, HAR, USPS, Pendigits, Reuters-10K, Coil100)



## License

Deep Clustering and Visualization (DCV) is released under the MIT license.