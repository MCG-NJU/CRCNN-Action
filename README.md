# Context-aware RCNNs: a Baseline for Action Detection in Videos
Source code for the following paper([arXiv link](https://arxiv.org/abs/2007.09861)):

    Context-aware RCNNs: a Baseline for Action Detection in Videos
    Jianchao Wu, Zhanghui Kuang, Limin Wang, Wayne Zhang, Gangshan Wu
    in ECCV 2020

Our implementation is based on [Video-long-term-feature-banks](https://github.com/facebookresearch/video-long-term-feature-banks).

## Prepare dataset
Please follow [LFB](https://github.com/facebookresearch/video-long-term-feature-banks/blob/master/DATASET.md) on how to prepare AVA dataset.


## Prepare environment
Please follow [LFB](https://github.com/facebookresearch/video-long-term-feature-banks/blob/master/INSTALL.md) on how to prepare Caffe2 environment.

## Download pre-trained weights
Please download [R50-I3D-NL](https://dl.fbaipublicfiles.com/video-long-term-feature-banks/r50_k400_pretrained.pkl), and put it in [code root]/pretrained_weights folder.

## Train a baseline model without scene feature and long-term feature
Run:
```bash
bash train_baseline.sh configs/avabox_r50_baseline_32x2_scale1_5.yaml
```

## Train a model with scene feature
Run:
```bash
bash train_baseline.sh configs/avabox_r50_baseline_16x4_scale1_5_withScene.yaml
```


## Train a model with scene feature and long-term feature
Stage1. Train a baseline model that will be used to infer LFB:
```bash
bash train_baseline.sh configs/avabox_r50_baseline_16x4_scale1_5.yaml
```

Stage2. Train a model with scene feature and LFB:
```bash
bash train_lfb.sh configs/avabox_r50_lfb_win60_L3_16x4_withScene.yaml [path to baseline model weight from step1]
```
