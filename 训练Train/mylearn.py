
import zipfile

import os
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import matplotlib
import sys
import pylab
import cv2

pylab.rcParams['figure.figsize'] = (8.0, 10.0)


from pycocotools.coco import COCO
root = 'F:\deeplearning\COCODataSet'  # 你下载的 COCO 数据集所在目录
dataType = 'val2017'
annFile = '{}/annotations/instances_{}.json'.format(root, dataType)


coco=COCO(annFile)  #读入标签
# display COCO categories and supercategories

cats = coco.loadCats(coco.getCatIds())
nms=[cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' '.join(nms)))

nms = set([cat['supercategory'] for cat in cats])
print('COCO supercategories: \n{}'.format(' '.join(nms)))

# 找到符合'person','dog','skateboard'过滤条件的category_id
catIds = coco.getCatIds(catNms=['person','dog','skateboard'])
# 找出符合category_id过滤条件的image_id
# imgIds = coco.getImgIds(catIds=catIds )
# 找出imgIds中images_id为324158的image_id
imgIds = coco.getImgIds(imgIds = [324158])
# 加载图片，获取图片的数字矩阵
img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]
# # 显示图片
#
# # I = io.imread(img['coco_url'])
I = io.imread('%s/%s/%s' % (root, dataType, img['file_name']))

# 语义分割识别
plt.imshow(I)
plt.axis('off')
annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
anns = coco.loadAnns(annIds)
coco.showAnns(anns)
plt.show()

#
# # 载入人体关键点标注
# annFile = '{}/annotations/person_keypoints_{}.json'.format(root, dataType)
# coco_kps = COCO(annFile)
#
# plt.imshow(I)
# plt.axis('off')
# ax = plt.gca()
# annIds = coco_kps.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
# anns = coco_kps.loadAnns(annIds)
# coco_kps.showAnns(anns)
# plt.show()

