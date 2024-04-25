# e-c2st
E-value Classifier Two-Sample Test

Note that the code structure copies [deep-anytime-testing](https://github.com/tpandeva/deep-anytime-testing).
This code will be incorporated into [deep-anytime-testing](https://github.com/tpandeva/deep-anytime-testing) soon.
Please refer to the deep-anytime-testing repository for more details.
## Setup the environment and install packages
```
python3 -m venv my_env #  conda create --name my_env (if you use conda)
source my_env/bin/activate # conda activate my_env
pip install -r requirements.txt
```
We also recommend to install wandb for logging. See https://docs.wandb.ai/quickstart for more details. We use wandb for logging the training process and storing the test statistics.

## Datasets
   **Blob Dataset**: The Blob dataset is a two-dimensional Gaussian mixture model with nine modes arranged on a 3 x 3 grid.  
   
**MNIST**: The MNIST dataset consists of hand-written digits. The DCGAN generated images are taken from: https://github.com/fengliu90/DK-for-TST

**KDEF**: Download the data from: https://kdef.se/download-2/ 
Separate the data into positive and negative images and store them in "data/positive" and "data/negative" folders.

## Running the code
1. Blob Dataset
```
python train.py experiment=blob-two-sample-e-c2st  # runs e-c2st 
python train.py experiment=blob-two-sample-baselines  # runs the baselines
```
2. MNIST Dataset
```
python train.py experiment=dcgan-mnist-cnn-e-c2st # runs e-c2st
python train.py experiment=dcgan-mnist-cnn-baselines # runs the baselines
```
3. KDEF Dataset
```
python train.py experiment=kdef-2st data.is_sequential=true # runs e-c2st
python train.py experiment=kdef-2st data.is_sequential=false # runs the baselines
```
run.ssh is a script that can be used to run the code on a remote server. 

