import os
import numpy as np
import cv2

# 首先遍历数据集，以image的时间戳为基准，将GroundTruth、class插值后与Image对齐
def align(image_path, gt_path, cls_path, new_gt_path, new_cls_path):
    gt_ts_list = []
    cls_ts_list = []
    image_ts_list = []
    gt_list = []
    cls_list=[]
    gt_paths = os.listdir(gt_path)
    gt_paths.sort(key=lambda x:float(x[:-4]))
    # 遍历数据集
    for cnt in gt_paths:
        gt = np.load(os.path.join(gt_path, cnt))
        cls = np.load(os.path.join(cls_path, cnt))
        gt_list.append(gt)
        cls_list.append(cls)
        gt_timestamp = cnt.split('/')[-1].split('.n')[0]
        gt_timestamp = float(gt_timestamp)
        gt_ts_list.append(gt_timestamp)
        cls_timestamp = cnt.split('/')[-1].split('.n')[0]
        cls_timestamp = float(cls_timestamp)
        cls_ts_list.append(cls_timestamp)

    image_paths = os.listdir(image_path)
    image_paths.sort(key=lambda x:float(x[:-4]))
    for image in image_paths:
        image_timestamp = image.split('/')[-1].split('.p')[0]
        image_timestamp = float(image_timestamp)
        image_ts_list.append(image_timestamp)

    print(len(gt_ts_list), len(cls_ts_list), len(image_ts_list))
    gt_list = np.array(gt_list)
    print(gt_list.size)

    #插值
    gt = np.zeros((len(image_ts_list), 3))
    gt[:,0] = np.interp(image_ts_list, gt_ts_list, gt_list[:,0])
    gt[:,1] = np.interp(image_ts_list, gt_ts_list, gt_list[:,1])
    gt[:,2] = np.interp(image_ts_list, gt_ts_list, gt_list[:,2])
    print(gt.size)
    cls_list = np.array(cls_list)
    cls = np.interp(image_ts_list, cls_ts_list, cls_list[:,0])
    
    # 保存对齐后的数据，分别到新的.npy文件中 
    for idx,i in enumerate(image_paths):
        i = i.split('/')[-1].split('.p')[0]
        np.save(os.path.join(new_gt_path,i+'.npy'), gt[idx])
        np.save(os.path.join(new_cls_path,i+'.npy'), cls[idx])
    return 0


def main():
    root_dir = './data/Anti_UAV_data/train'
    for i in range(102):
        seq_path =  os.path.join(root_dir,'seq'+str(i+1))
        print(seq_path)
        class_path = os.path.join(seq_path, 'class')
        image_path = os.path.join(seq_path, 'Image')
        gt_path = os.path.join(seq_path, 'ground_truth')
        if not os.path.exists(os.path.join(seq_path, 'gt')):
            os.mkdir(os.path.join(seq_path, 'gt'))
        if not os.path.exists(os.path.join(seq_path, 'cls')):
            os.mkdir(os.path.join(seq_path, 'cls'))
        new_gt_path = os.path.join(seq_path, 'gt')
        new_cls_path = os.path.join(seq_path, 'cls')
        align(image_path, gt_path, class_path, new_gt_path, new_cls_path)
        

        


if __name__ == '__main__':
    main()
