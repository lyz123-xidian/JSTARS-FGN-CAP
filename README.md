Feature guide network with Context Aggregation Pyramid for Remote Sensing Image Segmentation, JSTARS, 2022.
==
[Jiaojiao Li](https://scholar.google.com/citations?user=Ccu3-acAAAAJ&hl=zh-CN&oi=sra), [Yuzhe Liu](https://github.com/lyz123-xidian), Jiacha Liu, [Rui song](https://scholar.google.com/citations?user=_SKooBYAAAAJ&hl=zh-CN), Wei Liu, Kailiang Han and [Qian Du](https://scholar.google.com/citations?user=0OdKQoQAAAAJ&hl=zh-CN).
- Feature Guide Network With Context Aggregation Pyramid for Remote Sensing Image Segmentation, IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 2022, J. Li, Y. Liu, J. Liu, R. Song, W. Liu, K. Han and Q. Du, doi: 10.1109/JSTARS.2022.3221860.[[PDF]](https://ieeexplore.ieee.org/document/9947207)[[Code]](https://github.com/lyz123-xidian/TGRS-Sal2RN)
***
Code for the paper: [Feature guide network with Context Aggregation Pyramid for Remote Sensing Image Segmentation](https://ieeexplore.ieee.org/document/9947207).

<div align=center><img src="/Image/fig.png" width="80%" height="80%"></div>
Fig. 1: Our proposed network architecture. The network is based on encoder-decoder structure. In the encoder stage, edge and body information in the feature maps are extracted respectively constrained by corresponding loss function. The backbone is followed by a CAP to aggregate multi-scale context features adaptively. In the decoder stage, an EGFTM is used to help with upsampling to restore resolution.

Training and Test Process
--
1) The data sets used in this article are Vaihingen, Potsdam, TianZhi, and Cityscapes.
2) The network model file is placed in the 'models/net_work/final' folder and contains the ablation experiment network for each part.
3) Run "train.py" to retrain the FGN-CAP results.
4) Run "eval.py" to obtain the segmentation results of the training model
5) "train_cityscapes.py" and "test_cityscape.py" are training and testing code designed against the cityscape dataset.

We have successfully tested it on Ubuntu 18.04 with PyTorch 1.12.0.

References
--
If you find this code helpful, please kindly cite:

[1]J. Li et al., "Feature Guide Network With Context Aggregation Pyramid for Remote Sensing Image Segmentation," in IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 15, pp. 9900-9912, 2022, doi: 10.1109/JSTARS.2022.3221860.

Citation Details
--
BibTeX entry:
```
@article{li2022feature,
  title={Feature Guide Network With Context Aggregation Pyramid for Remote Sensing Image Segmentation},
  author={Li, Jiaojiao and Liu, Yuzhe and Liu, Jiachao and Song, Rui and Liu, Wei and Han, Kailiang and Du, Qian},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
  volume={15},
  pages={9900--9912},
  year={2022},
  publisher={IEEE}
}
```
 
Licensing
--
Copyright (C) 2022 Yuzhe Liu

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, version 3 of the License.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program.
