# 截止2020/7/24 #
# 路径 #
/detector/model：存放训练好的模型文件与Tensorboard记录
其中：
(Water1) 注水V1表示使用PS获得的水下WaterCo数据集训练结果
(Water2) 注水V2表示使用OpenCV获得的水下WaterCo数据集训练结果
注水V3表示使用Matlab获得的水下WaterCo数据集训练结果

均各训练20Epochs与40Epochs

# 模型文件 #

链接：https://pan.baidu.com/s/173ya4-s4W7izW57gEAVI0g 
提取码：qgc9


# 服务器训练方法 #

1. 开机
2. 进入PycharmProjects/TACO中，分别有DATA1-Training/DATA2-Training，分别对应注水V1、注水V2数据集
3. 进入detector文件夹，打开终端，激活环境
		source activate hrj-3.7
4. 打开detector.py，按照提示开始训练

<p align="center">
<img src="https://raw.githubusercontent.com/wiki/pedropro/TACO/images/logonav.png" width="25%"/>
</p>


TACO is a growing image dataset of waste in the wild. It contains images of litter taken under
diverse environments: woods, roads and beaches. These images are manually labeled and segmented
according to a hierarchical taxonomy to train and evaluate object detection algorithms. Currently,
images are hosted on Flickr and we have a server that is collecting more images and
annotations @ [tacodataset.org](http://tacodataset.org)


<div align="center">
  <div class="column">
    <img src="https://raw.githubusercontent.com/wiki/pedropro/TACO/images/1.png" width="17%" hspace="3">
    <img src="https://raw.githubusercontent.com/wiki/pedropro/TACO/images/2.png" width="17%" hspace="3">
    <img src="https://raw.githubusercontent.com/wiki/pedropro/TACO/images/3.png" width="17%" hspace="3">
    <img src="https://raw.githubusercontent.com/wiki/pedropro/TACO/images/4.png" width="17%" hspace="3">
    <img src="https://raw.githubusercontent.com/wiki/pedropro/TACO/images/5.png" width="17%" hspace="3">
  </div>
</div>
</br>

For convenience, annotations are provided in COCO format. Check the metadata here:
http://cocodataset.org/#format-data

TACO is still relatively small, but it is growing. Stay tuned!

# Publications

For more details check our paper: https://arxiv.org/abs/2003.06975

If you use this dataset and API in a publication, please cite us using: &nbsp;
```
@article{taco2020,
    title={TACO: Trash Annotations in Context for Litter Detection},
    author={Pedro F Proença and Pedro Simões},
    journal={arXiv preprint arXiv:2003.06975},
    year={2020}
}
```

# News
**December 20, 2019** - Added more 785 images and 2642 litter segmentations. <br/>
**November 20, 2019** - TACO is officially open for new annotations: http://tacodataset.org/annotate

# Getting started

### Requirements 

To install the required python packages simply type
```
pip3 install -r requirements.txt
```
Additionaly, to use ``demo.pynb``, you will also need [coco python api](https://github.com/cocodataset/cocoapi). You can get this using
```
pip3 install git+https://github.com/philferriere/cocoapi.git
```

### Download

To download the dataset images simply issue
```
python3 download.py
```
Alternatively, download from [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3242156.svg)](https://doi.org/10.5281/zenodo.3242156) 

Our API contains a jupyter notebook ``demo.pynb`` to inspect the dataset and visualize annotations.

### Trash Detection

The implementation of [Mask R-CNN by Matterport](https://github.com/matterport/Mask_RCNN)  is included in ``/detector``
with a few modifications. Requirements are the same. Before using this, the dataset needs to be split. You can either donwload our [weights and splits](https://github.com/pedropro/TACO/releases/tag/1.0) or generate these from scratch using the `split_dataset.py` script to generate 
N random train, val, test subsets. For example, run this inside the directory `detector`:
```
python3 split_dataset.py --dataset_dir ../data
```

For further usage instructions, check ``detector/detector.py``.

As you can see [here](http://tacodataset.org/stats), most of the original classes of TACO have very few annotations, therefore these must be either left out or merged together. Depending on the problem, ``detector/taco_config`` contains several class maps to target classes, which maintain the most dominant classes, e.g., Can, Bottles and Plastic bags. Feel free to make your own classes.

<p align="center">
<img src="https://raw.githubusercontent.com/wiki/pedropro/TACO/images/teaser.gif" width="75%"/></p>
