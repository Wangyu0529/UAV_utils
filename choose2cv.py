#读取npy文件的x,y,z坐标，如果x,y,z欧式距离小于20，则挑选出来，组成新的数据集
import os
import numpy as np
import cv2

def choose2cv(gt_path):
    gt_list = os.listdir(gt_path)
    gt_list.sort(key=lambda x:float(x[:-4]))
    chosen_gt = []
    for i in range(len(gt_list)):
        gt = np.load(os.path.join(gt_path, gt_list[i]))
        #把gt的维度求欧式距离
        distance = np.sqrt(np.sum(gt**2))
        if distance < 10:
            chosen_gt.append(gt_list[i])
    print(len(chosen_gt))
    return chosen_gt

def composition(gt_path):
    #遍历整个文件夹，根据gt距离，分类出10m、20m、30m、30m以上的文件数量
    gt_list = os.listdir(gt_path)
    gt_list.sort(key=lambda x:float(x[:-4]))
    gt_10 = 0
    gt_20 = 0
    gt_30 = 0
    gt_30p = 0
    for i in range(len(gt_list)):
        gt = np.load(os.path.join(gt_path, gt_list[i]))
        distance = np.sqrt(np.sum(gt**2))
        if distance < 10:
            gt_10 += 1
        elif distance < 20:
            gt_20 += 1
        elif distance < 30:
            gt_30 += 1
        else:
            gt_30p += 1
    return gt_10, gt_20, gt_30, gt_30p
    
    


def copy2cv(chosen_gt, root_dir, seq_dir):
    image_path = os.path.join(seq_dir, 'Image')
    image_list = os.listdir(image_path)
    image_list.sort(key=lambda x:float(x[:-4]))
    cv_dir = os.path.join(root_dir, 'images')
    if not os.path.exists(cv_dir):
        os.makedirs(cv_dir)
    for i in range(len(chosen_gt)):
        if chosen_gt[i].replace('.npy','.png') in image_list:
            img = cv2.imread(os.path.join(image_path, chosen_gt[i].replace('.npy','.png')))
            cv2.imwrite(os.path.join(cv_dir, chosen_gt[i].replace('.npy','.png')), img)
    return 0
        



def main():
    root_dir = './data/Anti_UAV_data/train'
    gt_10, gt_20, gt_30, gt_30p = 0, 0, 0, 0
    for i in range(len(os.listdir(root_dir))):
        seq_path =  os.path.join(root_dir,'seq'+str(i+1))
        print(seq_path)
        gt_path = os.path.join(seq_path, 'gt')
        # chosen = choose2cv(gt_path)
        # copy2cv(chosen, root_dir, seq_path)
        gt_10 += composition(gt_path)[0]
        gt_20 += composition(gt_path)[1]
        gt_30 += composition(gt_path)[2]
        gt_30p += composition(gt_path)[3]
    total = gt_10 + gt_20 + gt_30 + gt_30p
    percentage_10 = gt_10 / total
    percentage_20 = gt_20 / total
    percentage_30 = gt_30 / total
    percentage_30p = gt_30p / total
    print(gt_10, percentage_10)
    print(gt_20, percentage_20)
    print(gt_30, percentage_30)
    print(gt_30p, percentage_30p)
        
if __name__ == '__main__':
    main()