import os
import sys
import numpy as np
import json
import cv2
from torch.utils.data import Dataset, DataLoader


class MTCDataset(Dataset):
    def __init__(self,
                 root_dir,
                 istrain=True,
                 transform=None
                  ):
        # choose which dataset, train or val
        if istrain:
            self.root_dir = os.path.join(root_dir, 'train')
        else:
            self.root_dir = os.path.join(root_dir, 'val')

        # get class/gt/image/lidar/livox address
        self.cnt_list = []
        self.image_cnt_list = []
        self.lidar_cnt_list = []
        seqs = os.listdir(self.root_dir)
        for seq in seqs:
            seq_paths = os.path.join(self.root_dir, seq)
            seq_list = os.listdir(seq_paths)
            class_paths = os.path.join(seq_paths, 'class')
            cnts =  os.listdir(class_paths)
            for cnt in cnts:
                self.cnt_list.append([seq, cnt])
            image_paths = os.path.join(seq_paths,'Image')
            image_cnts = os.listdir(image_cnts)
            for cnt in image_cnts:
                self.image_cnt_list.append([seq, cnt])
            lidar360_paths = os.path.join(seq_paths,'lidar_360')
            lidar_cnts = os.listdir(lidar_cnts)
            for cnt in lidar_cnts:
                self.lidar_cnt_list.append([seq, cnt])
    

    # load and postprocess the data
    def _load_class(self, class_dir, idx):
        cl = np.interp(image)
        return cl

    def _load_image(self, image_dir):
        image_dir = image_dir.replace('.npy', '.png')
        image = cv2.imread(image_dir)
        return image

    #根据时间戳将GroundTruth插值后，与Image对齐
    def gt_image_align(self,gt_dir):
        for i in os.listdir(gt_dir):
            item = np.load(os.path.join(gt_dir,i))
            gt = np.vstack((gt,item))
        gt_timestamp = self.cnt_list[1].split('.')[0]
        gt_timestamp = int(gt_timestamp)
        image_timestamp = self.image_cnt_list[1].split('.')[0]
        image_timestamp = int(image_timestamp)
        gt = np.interp(image_timestamp, gt_timestamp, gt)
        return gt

    def _load_gt(self, gt, idx):    
        return gt[idx]
    


    def __getitem__(self, idx):
        class_path = os.path.join(self.root_dir,self.cnt_list[idx][0],'class',self.cnt_list[idx][1])
        gt_path = os.path.join(self.root_dir,self.cnt_list[idx][0],'ground_truth')
        image_path = os.path.join(self.root_dir,self.image_cnt_list[idx][0],'Image',self.image_cnt_list[idx][1])
        lidar_path = os.path.join(self.root_dir,self.lidar_cnt_list[idx][0],'Image',self.lidar_cnt_list[idx][1])
        cl = self._load_class(class_path,idx)
        image = self._load_image(image_path)
        gt_list = self.gt_image_align(gt_path)
        gt = self._load_gt(gt_list,idx)
        return cl, image, gt


    def __len__(self):
        return len(self.image_cnt_list)


def main():
    MTC = MTCDataset('./data/Anti_UAV_data')
    trainloader = DataLoader(MTC, batch_size=4,shuffle=True)
    for i, batch in enumerate(trainloader):
        print(cl.size,image.size,gt.size)

if __name__ == '__main__':
    main()
