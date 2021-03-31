# Occlusion Guided Self-supervised Scene Flow Estimation on 3D Point Clouds
This is the official implementation for the paper "Occlusion Guided Self-supervised Scene Flow Estimation on 3D Point Clouds"

## Requirement
To run our model (3D-OGFlow), please install the following package (we suggest to use the Anaconda environment):
* Python 3.6+
* PyTorch==1.6.0
* CUDA CuDNN
* Pytorch-lightning==1.1.0
* numpy
* tqdm

Compile the furthest point sampling, grouping and gathering operation for PyTorch. We use the operation from this [repo](https://github.com/sshaoshuai/Pointnet2.PyTorch).
```shell
cd pointnet2
python setup.py install
cd ../
```

## Data preperation
We use the Flyingthings3D and KITTI dataset preprocessed by [this work](https://github.com/xingyul/flownet3d).
Download the Flyingthings3D dataset from [here](https://drive.google.com/file/d/1CMaxdt-Tg1Wct8v8eGNwuT7qRSIyJPY-/view?usp=sharing) and KITTI dataset from [here](https://drive.google.com/open?id=1XBsF35wKY0rmaL7x7grD_evvKCAccbKi).
 Create a folder named `datasets` under the root folder. After the downloading, extract the files into the `datasets`. The directory of the datasets should looks like the following:

```
datasets/data_processed_maxcut_35_20k_2k_8192   % FlyingThings3D dataset
datasets/kitti_rm_ground                        % KITTI dataset
```

## Get started

### Supervised Training
In order to train our model on the Flyingthings3D dataset with the supervised loss, run the following:

```bash
$ python train.py --num_points 8192 --batch_size 8 --epochs 120 --use_multi_gpu True
```
for the help on how to use the optional arguments, type:
```bash
$ python train.py --help
```

### Self-supervised Training
In order to train our model on the Flyingthings3D dataset by using our proposed self-supervised scheme, run the following:

```bash
$ python train_self_ln.py --num_points 8192 --batch_size 3 --epochs 150 --use_multi_gpu True
```
for the help on how to use the optional arguments, type:
```bash
$ python train.py --help
```
Notice that in order to speed up the running time and to have a better utilization of the GPUs, our self-supervised training code is implemented using the [PyTorch Lightning](https://www.pytorchlightning.ai/) library.


### Evaluation
We provide two pretained weights of 3D-OGFlow, one from the supervised training and the other from the self-supervised training. In order to evaluate our pretrained model under the ```pretrained_model``` folder with the Flyingthings3D dataset, run the following:

```bash
$ python evaluate.py --num_points 8192 --dataset f3d --ckp_path ./pretrained_model/supervised/PointPWOC_88.6285_114_0.1409.pth
```

for the evaluation on KITTI dataset, run the following:
```bash
$ python evaluate.py --num_points 8192 --dataset kitti --ckp_path ./pretrained_model/supervised/PointPWOC_88.6285_114_0.1409.pth
```
For help on how to use this script, type:
```bash
$ python evaluate.py --help
```
