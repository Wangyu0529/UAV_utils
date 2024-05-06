# 根据gt，给出lidar的label
# 格式为：[x,y,z,dx,dy,dz,heading_angle, category_name]
import os
import numpy as np



def lidarBox(gt_dir, cls_dir, target_dir):
    dx = 4
    dy = 4
    dz = 4
    gt_list = os.listdir(gt_dir)
    gt_list.sort(key=lambda x:float(x[:-8]))
    for i in range(len(gt_list)):
        gt = np.load(os.path.join(gt_dir, gt_list[i]))
        cl = np.load(os.path.join(cls_dir, gt_list[i]))
        print(cl)
        # 计算heading_angle
        heading_angle = np.arctan2(gt[1], gt[0])
        # 根据gt，给出lidar的label
        # 格式为：[x,y,z,dx,dy,dz,heading_angle, category_name]
        label = [gt[0],gt[1],gt[2],dx,dy,dz, heading_angle, cl]
        print(label)
        with open( os.path.join(target_dir, 'labels',gt_list[i].replace('.npy.npy','.txt')),'w') as f:
            f.write(' '.join([str(x) for x in label]))

# 读取list文件，返回txt
def getIndex(target_dir):
    label_list = os.listdir(os.path.join(target_dir, 'labels'))
    label_list.sort(key=lambda x:float(x[:-4]))
    with open(os.path.join(target_dir, 'train.txt'),'w') as f:
        for label in label_list:
            f.write(os.path.join(target_dir, 'labels', label.replace('.txt',''))+'\n')
        

if __name__ == '__main__':
    for i in range(102):
        root_dir = './data/Anti_UAV_data/train'
        target_dir = './OpenPCDet/data/custom'
        root_dir = os.path.join(root_dir, 'seq'+str(i+1))
        gt_dir = os.path.join(root_dir, 'gt_lidar')
        cls_dir = os.path.join(root_dir, 'cls_lidar')
        # lidarBox(gt_dir, cls_dir, target_dir)
        # 把lidar360数据搬到target_dir
        lidar360_dir = os.path.join(root_dir, 'lidar_360')
        lidar360_list = os.listdir(lidar360_dir)
        lidar360_list.sort(key=lambda x:float(x[:-4]))
        for j in range(len(lidar360_list)):
            lidar360 = np.load(os.path.join(lidar360_dir, lidar360_list[j]))
            np.save(os.path.join(target_dir, 'points', lidar360_list[j]), lidar360)