import os
import numpy as np

root_dir = "../data/custom"
lidar_dir = os.path.join(root_dir, 'points')
lidar_list = os.listdir(lidar_dir)
lidar_list.sort(key=lambda x:float(x[:-4]))
gt_dir = os.path.join(root_dir, 'labels')
for i in range(len(lidar_list)):
    gt = [0, 0, 0, 4, 4, 4, 0, 1]
    with open(os.path.join(gt_dir, lidar_list[i].replace('.npy','.txt')),'w') as f:
        f.write(' '.join([str(x) for x in gt]))

