# Channel Augmented Joint Learning for Visible-Infrared Recognition (ICCV 2021) 
Pytorch Code of CAJ method [1] for Cross-Modality Person Re-Identification (Visible Thermal Re-ID) on RegDB dataset [4] and SYSU-MM01 dataset [5]. 

A Huawei MindSpore implementation of our proposed method is avaiable at [MindSpore CAJ](https://gitee.com/mindspore/contrib/tree/master/papers/CAJ).

We adopt the two-stream network structure introduced in [2,3].

|Datasets    | Pretrained| Rank@1  | mAP |  mINP |  Model|
| --------   | -----    | -----  |  -----  | ----- |------|
|#RegDB      | ImageNet | ~ 85.03% | ~ 79.14%|  ~65.33% |----- |
|#SYSU-MM01  | ImageNet | ~ 69.88%  | ~ 66.89% | ~53.61% | [GoogleDrive](https://drive.google.com/file/d/1vIKkB61frqA-zG0RiL282heqthvwkKdO/view?usp=sharing)|

*Both of these two datasets may have some fluctuation due to random spliting. The results might be better by finetuning the hyper-parameters. 

### 1. Prepare the datasets.

- (1) RegDB Dataset [3]: The RegDB dataset can be downloaded from this [website](http://dm.dongguk.edu/link.html) by submitting a copyright form.

- (Named: "Dongguk Body-based Person Recognition Database (DBPerson-Recog-DB1)" on their website). 

- A private download link can be requested via sending me an email (mangye16@gmail.com). 

- (2) SYSU-MM01 Dataset [4]: The SYSU-MM01 dataset can be downloaded from this [website](http://isee.sysu.edu.cn/project/RGBIRReID.htm).

- run `python pre_process_sysu.py` to pepare the dataset, the training data will be stored in ".npy" format.

### 2. Joint Training (Section 4.2).
Train a model by
```bash
python train_ext.py --dataset sysu --lr 0.1 --method adp --augc 1 --rande 0.5 --alpha 1 --square 1 --gamma 1 --gpu 1
```

- `--dataset`: which dataset "sysu" or "regdb".

- `--lr`: initial learning rate.

-  `--method`: method to run Enhanced Squared Difference or Baseline.

-  `--augc`:  Channel augmentation or not.

-  `--rande`:  random erasing with probability.

- `--gpu`:  which gpu to run.

You may need mannully define the data path first.

**Parameters**: More parameters can be found in the script.

**Sampling Strategy**: N (= bacth size) person identities are randomly sampled at each step, then randomly select four visible and four thermal image. Details can be found in Line 302-307 in `train.py`.

**Training Log**: The training log will be saved in `log/" dataset_name"+ log`. Model will be saved in `save_model/`.

### 3. Testing.

Test a model on SYSU-MM01 or RegDB dataset by using testing augmentation with HorizontalFlip
```bash
python testa.py --mode all --resume 'model_path' --gpu 1 --dataset sysu
```
- `--dataset`: which dataset "sysu" or "regdb".

- `--mode`: "all" or "indoor" all search or indoor search (only for sysu dataset).

- `--trial`: testing trial (only for RegDB dataset).

- `--resume`: the saved model path.

- `--gpu`:  which gpu to run.

### 4. Citation

Please kindly cite this paper in your publications if it helps your research:
```
@inproceedings{iccv21caj,
author    = {Ye, Mang and Ruan, Weijian and Du, Bo and Shou, Mike Zheng},
title     = {Channel Augmented Joint Learning for Visible-Infrared Recognition},
booktitle = {IEEE/CVF International Conference on Computer Vision},
year      = {2021},
pages     = {13567-13576}
}
```

###  5. References.

[1] M. Ye, W. Ruan, B. Du, and M. Shou. Channel Augmented Joint Learning for Visible-Infrared Recognition. IEEE International Conference on Computer Vision (ICCV), 2021.

[2] M. Ye, J. Shen, G. Lin, T. Xiang, L. Shao, and S. C., Hoi. 	Deep learning for person re-identification: A survey and outlook. IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2021.

[3] M. Ye, X. Lan, Z. Wang, and P. C. Yuen. Bi-directional Center-Constrained Top-Ranking for Visible Thermal Person Re-Identification. IEEE Transactions on Information Forensics and Security (TIFS), 2019.

[4] D. T. Nguyen, H. G. Hong, K. W. Kim, and K. R. Park. Person recognition system based on a combination of body images from visible light and thermal cameras. Sensors, 17(3):605, 2017.

[5] A. Wu, W.-s. Zheng, H.-X. Yu, S. Gong, and J. Lai. Rgb-infrared crossmodality person re-identification. In IEEE International Conference on Computer Vision (ICCV), pages 5380–5389, 2017.

Contact: mangye16@gmail.com
