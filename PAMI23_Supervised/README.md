# Channel Augmentation for Visible-Infrared Re-Identification 
Pytorch Code of CAJ+ method (Section 4.2) of Channel Augmentation for Visible-Infrared Re-Identification (PAMI 2023). 

*Both of these two datasets may have some fluctuation due to random spliting. The results might be better by finetuning the hyper-parameters. 

### 1. Prepare the datasets.

- (1) RegDB Dataset [3]: The RegDB dataset can be downloaded from this [website](http://dm.dongguk.edu/link.html) by submitting a copyright form.

- (Named: "Dongguk Body-based Person Recognition Database (DBPerson-Recog-DB1)" on their website). 

- A private download link can be requested via sending me an email (mangye16@gmail.com). 

- (2) SYSU-MM01 Dataset [4]: The SYSU-MM01 dataset can be downloaded from this [website](http://isee.sysu.edu.cn/project/RGBIRReID.htm).

- run `python pre_process_sysu.py` to pepare the dataset, the training data will be stored in ".npy" format.

### 2. Weak-and-Strong Augmentation Joint Training (Section 4.2).
Train a model by
```bash
python3 train_ext.py --dataset sysu --lr 0.1 --method adp --augc 1 --rande 0.5 --alpha 1 --square 1 --gamma 1 --gpu 1
```


### 3. Testing.

Test a model on SYSU-MM01 or RegDB dataset by using testing augmentation with HorizontalFlip
```bash
python3 testa.py --mode all --resume 'model_path' --gpu 1 --dataset sysu
```
- `--dataset`: which dataset "sysu" or "regdb".

- `--mode`: "all" or "indoor" all search or indoor search (only for sysu dataset).

- `--trial`: testing trial (only for RegDB dataset).

- `--resume`: the saved model path.

- `--gpu`:  which gpu to run.

### 4. Citation

Please kindly cite this paper in your publications if it helps your research:
```
@article{ye2023channel,
  title={Channel Augmentation for Visible-Infrared Re-Identification.},
  author={Ye, Mang and Wu, Zesen and Chen, Cuiqun and Du, Bo},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2023}
}
```

###  5. Concat.

The code is based on [Channel Augmentation](https://github.com/mangye16/Cross-Modal-Re-ID-baseline/tree/master/ICCV21_CAJ).

Contact: chencuiqun@whu.edu.cn
