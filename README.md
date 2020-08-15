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

## Train a baseline model
Run:
```bash
bash train_baseline.sh configs/avabox_r50_baseline_32x2_scale1_5.yaml
```


