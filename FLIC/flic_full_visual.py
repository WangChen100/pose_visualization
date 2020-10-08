import os
import numpy as np
import json
import cv2
import random
from tqdm import tqdm
from scipy.io import loadmat
import math

def show_skeleton(img,kpts,color=(255,128,128)):
    skelenton = [[0, 1], [1, 2], [3, 4], [4, 5],
                 [13, 12], [12,16], [13,16],
                 [6,9]]
    # skelenton = [[10, 11], [11, 12], [12, 8], [8, 13], [13, 14], [14, 15], [8, 9], [7, 8], [2, 6],
    #              [3, 6], [1, 2], [1, 0], [3, 4], [4, 5],[6,7]]
    points_num = [num for num in range(1, 29)]
    for sk in skelenton:
        if math.isnan(kpts[sk[0]][0]): continue
        if math.isnan(kpts[sk[1]][0]): continue
        pos1 = (int(kpts[sk[0]][0]), int(kpts[sk[0]][1]))
        pos2 = (int(kpts[sk[1]][0]), int(kpts[sk[1]][1]))
        if pos1[0] > 0 and pos1[1] > 0 and pos2[0] > 0 and pos2[1] > 0:
            cv2.line(img, pos1, pos2, color, 2, 8)
    for points in points_num:
        if math.isnan(kpts[points-1][0]): continue
        pos = (int(kpts[points-1][0]),int(kpts[points-1][1]))
        if pos[0] > 0 and pos[1] > 0 :
            cv2.circle(img, pos,3,(0,0,255),-1) #为肢体点画红色实心圆
    return img

m = loadmat("examples.mat")
examples=m['examples']
filepath = examples['filepath']
coords = examples['coords']


file_name = "american-wedding-unrated6x9-00003831.jpg" #"2-fast-2-furious-00003631.jpg"
image = cv2.imread(file_name)
skeleton_color = [(154, 194, 182),
                  (123, 151, 138),
                  (0,   208, 244),
                  (8,   131, 229),
                  (18,  87,  220)]  # 选择自己喜欢的颜色
for id in range(5003):
    if file_name == filepath[0][id][0]:
        print("yes")
        joints=coords[0][id].T
        show_skeleton(image, joints, color=(0,   208, 244))
        break
# cv2.namedWindow('lsp', cv2.WINDOW_NORMAL)
# cv2.imshow("lsp"+str(id), image)
# cv2.waitKey()
cv2.imwrite("flic_3.png", image)

