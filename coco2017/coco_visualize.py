from crowdposetools.coco import COCO  # pycocotools
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
pylab.rcParams['figure.figsize'] = (8.0, 10.0)

# dataDir= '/home/andrew/datasets/MSCOCO/coco2017'
# dataType='train2017'
# annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)
annFile='data/crowdpose/person_keypoints_train2017.json'
# 初始化标注数据的 COCO api
coco=COCO(annFile)

imgIds = coco.getImgIds(imgIds = [100000])
img = coco.loadImgs(imgIds[0])[0]

I = io.imread('data/crowdpose/%s'%(img['file_name']))
plt.figure()
plt.imshow(I)
plt.axis('off')

ax = plt.gca()
annIds = coco.getAnnIds(imgIds=img['id'], iscrowd=None)
anns = coco.loadAnns(annIds)
coco.showAnns(anns)
plt.show()
# plt.savefig('coco4.png')
