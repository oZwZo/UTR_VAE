# UTR_VAE

# Multi-task learning for sequence modeling.
- A: Intigrating multiple-dataset to learn common regularoty feature
- B : Embed multiple sequence characteristics into one joint space
![image](https://user-images.githubusercontent.com/46890438/141752186-6358d68b-a6a6-4f65-aae2-81c68e6a98dd.png)

# prerequistes
```shell
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
conda install pandas matplotlib seaborn configparser scikit-learn=0.23
conda install Biopython logomaker -c bioconda 
```

# Model Configuring  

click into any `ini` file under the `log/` directory, copy one and change the parameter in the file. Some **template configs** were provided for your reference.

# Training 
- for multi-dataset model:
  ```shell
  python script/iter_train.py --config_file  <CONFIG_PATH>
  ```
  - if you are running on a GPU feasible machine:
  ```shell
  python script/iter_train.py --config_file  <CONFIG_PATH> --CUDA <GPU DEVICE INDEX> --kfold_index <which CV fold>
  ```
- for softshared multi-task model :
  ```shell
  python script/main_train.py --config_file 
  ```
# 
