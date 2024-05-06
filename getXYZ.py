import numpy as np
import os


root_dir = './data/Anti_UAV_data'

seq_dict = {}
for i in range(102):    
    seq_x = 0.0
    seq_y = 0.0
    seq_z = 0.0
    cnt = 0
    seq_dir = os.path.join(root_dir, "train","seq"+str(i+1))
    gt_dir = os.path.join(seq_dir, 'ground_truth')
    gt_list = os.listdir(gt_dir)
    for j in gt_list:
        gt = np.load(os.path.join(gt_dir, j))
        x = gt[0]
        y = gt[1]
        z = gt[2]
        seq_x += x
        seq_y += y
        seq_z += z
        cnt += 1
    seq_dict["seq"+str(i+1)] = [seq_x/cnt, seq_y/cnt, seq_z/cnt]
for i in range(16):
    seq_x = 0.0
    seq_y = 0.0
    seq_z = 0.0
    cnt = 0
    seq_dir = os.path.join(root_dir, "val","seq"+str(i+1))
    gt_dir = os.path.join(seq_dir, 'ground_truth')
    gt_list = os.listdir(gt_dir)
    for j in gt_list:
        gt = np.load(os.path.join(gt_dir, j))
        x = gt[0]
        y = gt[1]
        z = gt[2]
        seq_x += x
        seq_y += y
        seq_z += z
        cnt += 1
    seq_dict["seq"+str(i+1)] = [seq_x/cnt, seq_y/cnt, seq_z/cnt]

for key in seq_dict.keys():
    print(key, seq_dict[key])