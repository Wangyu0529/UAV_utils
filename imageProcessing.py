import numpy as np
import os
import cv2
import sys
sys.path.append('./dataset')
from ultralytics import YOLO
from getGT import InversePM

model = YOLO('./runs/detect/0423/train4/weights/best.pt')
def output(image):
    # Load the model
    
    results = model(image)
    for result in results:
        boxes = result.boxes.cpu()  # Boxes object for bounding box outputs
        probs = result.probs  # Probs object for classification outputs
        xyxy = boxes.xyxy
        # 如果xyxy[0] size 0，则赋值
        if xyxy.size(0) == 0:
            xyxy = np.array([0,0,0,0])
        else:
            xyxy = xyxy[0].numpy()
    return  xyxy, probs

def angle(xyxy):
    center = [xyxy[0]+(xyxy[2]-xyxy[0])/2,xyxy[1]+(xyxy[3]-xyxy[1])/2]
    # 根据鱼眼相机几何关系，计算归一化三轴坐标
    target = InversePM(center)
    return target
    
def match(target, lidar_path):
    lidar_list = os.listdir(lidar_path)
    lidar_list.sort(key=lambda x:float(x[:-4]))
    for j in range(len(lidar_list)):
        lidar = np.load(os.path.join(lidar_path, lidar_list[j]))
        # 遍历lidar数据,如果lidar数据的某个点在target数据的范围内，则correct
        eps = 1e-6
        for i in range(len(lidar)):
            if lidar[i][0]/lidar[i][1]- (target[0]+eps)/(target[1]+eps)<0.1 and lidar[i][0]/lidar[i][2]- (target[0]+eps)/(target[2]+eps)<0.1:
                print(lidar[i])
                return lidar[i] 
    return [0,0,0,0]

def main():
    root_dir = './data/Anti_UAV_data/val'
    lidar_path = os.path.join(root_dir, 'lidar')
    image_path = os.path.join(root_dir, 'images')
    image_list = os.listdir(image_path)
    image_list.sort(key=lambda x:float(x[:-5]))
    results = []
    for i in range(len(image_list)):
        image = cv2.imread(os.path.join(image_path, image_list[i]))
        boxes,probs = output(image)
        # if not boxes[0] == 0 and boxes[1] == 0 and boxes[2] == 0 and boxes[3] == 0:
        target = angle(boxes)
        result = match(target, lidar_path)
        with open( os.path.join(root_dir, 'labels',image_list[i].replace('.png','.txt')),'r') as f:
            gt = f.readlines()
            gt = [x.strip().split(' ') for x in gt]
        if np.sum(result-gt) < 1:
            results.append(result)
    print(results)

if __name__ == '__main__':
    main()