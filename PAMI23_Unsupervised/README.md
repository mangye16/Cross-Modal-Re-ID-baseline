# Channel Augmentation for Visible-Infrared Re-Identification (Unsupervised Augmented Association)

Pytorch Code of Unsupervised Augmented Association (Section.5) for Channel Augmentation for Visible-Infrared Re-Identification (PAMI 2023).

## 0. Environment
Some key Python packages and their versions are listed as below:
```
cudatoolkit=11.1.74
faiss-gpu=1.6.4
python=3.6.13
scikit-learn=0.22.1
torch=1.7.0
torchvision=0.8.0
```
You may need to install other required relevant packages.

## 1. Prepare the dataset
* (1) The Dataset Structure of both SYSU-MM01 and RegDB dataset should be reorganized like this:
  * --bounding_box_test
  * --bounding_box_train
  * --query

* (2) Our prepared dataset could be downloaded from [Google Drive](https://drive.google.com/drive/folders/1jUPdNeMVTjcqiTm5RlcXvx7FAcDhuigN?usp=sharing).

## 2. Training
You may need to modify the dataset path and log path before training.
The parameter can be modified in the below shell script, where more details are avaiable.

Train a model for SYSU-MM01 (or RegDB) by
```
./run-sysu.sh
```
or
```
./run-regdb.sh
```

Please note that there are some parameters that are different for the two datasets, so please adjust them before training.

We give specific adjustments here:
1. `eps` for clustering (line 40 and line 41 in `reid\utils\clustering.py`)
2. `camid` (line 118 and line 121 in `reid\utils\data\preprocessor.py`)
3. `camid` (line 29 and line 33 in `reid\utils\clustering.py`)

Our trained model and log example for both two datasets can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1JA6drRHgDqTCYCIEWY1AoxQZb9iTHolx?usp=sharing).

## 3. Testing
You may also need to modify the dataset path and log path before testing. 

Test a model by 
```shell
./test-sysu.sh
```
or
```shell
./test-regdb.sh
```


## 4. Citation
Please kindly cite this paper in your publications if it helps your research:
```
@article{ye2023channel,
  title={Channel Augmentation for Visible-Infrared Re-Identification.},
  author={Ye, Mang and Wu, Zesen and Chen, Cuiqun and Du, Bo},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2023}
}
```

## 5. Contact

Please be free to contact with me (zesenwu@whu.edu.cn).

The code is implemented based on [Channel Augmentation](https://github.com/mangye16/Cross-Modal-Re-ID-baseline/tree/master/ICCV21_CAJ) and [Camera Aware Proxy](https://github.com/Terminator8758/CAP-master).