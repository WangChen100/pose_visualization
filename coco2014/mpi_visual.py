import os
import numpy as np
import json
import cv2
import random
from tqdm import tqdm
dataset_dir = "D:/send_paper/MPII/mpii_valid/mpii_valid/"
dataset_save = "D:/send_paper/MPII/result/"

def show_skeleton(img,kpts,color=(255,128,128)):
    skelenton = [[10, 11], [11, 12], [12, 8], [8, 13], [13, 14], [14, 15], [8, 9], [7, 8], [2, 6],
                 [3, 6], [1, 2], [1, 0], [3, 4], [4, 5],[6,7]]
    points_num = [num for num in range(1,16)]
    for sk in skelenton:
        pos1 = (int(kpts[sk[0]][0]), int(kpts[sk[0]][1]))
        pos2 = (int(kpts[sk[1]][0]), int(kpts[sk[1]][1]))
        if pos1[0] > 0 and pos1[1] > 0 and pos2[0] > 0 and pos2[1] > 0:
            cv2.line(img, pos1, pos2, color, 2, 8)
    for points in points_num:
        pos = (int(kpts[points-1][0]),int(kpts[points-1][1]))
        if pos[0] > 0 and pos[1] > 0 :
            cv2.circle(img, pos,4,(0,0,255),-1) #为肢体点画红色实心圆
    return img

with open("D:/send_paper/MPII/valid.json","r") as load_f:
    load_dict = json.load(load_f)
imgIds_old = "0.jpg"
image = cv2.imread(os.path.join(dataset_dir, '005808361.jpg'))
skeleton_color = [(154,194,182),(123,151,138),(0,208,244),(8,131,229),(18,87,220)] # 选择自己喜欢的颜色
for dict_num in tqdm(load_dict):
    imgIds = dict_num["image"]
    if imgIds!=imgIds_old:
        cv2.imwrite(os.path.join(dataset_save,imgIds_old),image)
        image_path = os.path.join(dataset_dir, imgIds)
        image = cv2.imread(image_path)
    color = random.choice(skeleton_color)
    show_skeleton(image, dict_num["joints"],color=color)

    imgIds_old = imgIds
