# PartDistill

[[Project Page](https://ardianumam.github.io/partdistill/)] [[Code](https://github.com/ardianumam/PartDistill)] [[Arxiv](https://arxiv.org/abs/2312.04016)] [[Video](https://www.youtube.com/watch?v=bYR2B7UndeM&t=171s&ab_channel=ArdianUmam)] [[Poster](https://drive.google.com/file/d/1DtQ5DuQFXqF2JtDciJ0qYeqCdxHDKrNH/view?usp=sharing)]

This is the official repository of our CVPR 2024 paper, "PartDistill: 3D Shape Part Segmentation by Vision-Language Model Distillation". 

<img src="assets/partdistill_animate.gif" width="600">

## Data
* The prapared data can be downloaded [here](https://drive.google.com/drive/folders/1kzugzkn_9dO-37GcnaQm2qmt4MO7Fw-B?usp=sharing). The content details are explained in [data_format.md](data_format.md)
* Point-M2AE model can be download [here](https://github.com/ZrrSkywalker/Point-M2AE?tab=readme-ov-file)

## Train
To train the model, please use script bellow
```
CUDA_VISIBLE_DEVICES=<gpu_idx> python train.py --data_path<dir_path_storing_preprocessed_data> \
 --ckpt_dir=<dir_to_store_ckpt> \
 --backbone_path=<path_to_pointm2ae_ckpt> \
 --lr=0.003 \
 --batch_size=16 \
 --category chair airplane knife \
 --n_epoch=25 \
 --exp_suffix=exp1
```
For example:
```
CUDA_VISIBLE_DEVICES=8 python train.py --data_path=/disk2/aumam/dataset/shapenet/shapenetcore_partanno_distill \
 --ckpt_dir=/disk2/aumam/dev/multimodal_distillation/checkpoints \
 --backbone_path=/disk2/aumam/dev/multimodal_distillation/checkpoints/point_m2ae_pre-train.pth \
 --lr=0.003 \
 --batch_size=16 \
 --category chair airplane knife \
 --n_epoch=25 \
 --exp_suffix=exp1
```

## Note
### Install this library to use Point-M2AE
```
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
```

## Cite
```
@inproceedings{umam2023partdistill,
  title = {PartDistill: 3D Shape Part Segmentation by Vision-Language Model Distillation},
  author = {Umam, Ardian and Yang, Cheng-Kun and Chen, Min-Hung and Chuang, Jen-Hui and Lin, Yen-Yu},
  booktitle = {IEEE/CVF International Conference on Computer Vision (CVPR)},
  year = {2024},
}
```

