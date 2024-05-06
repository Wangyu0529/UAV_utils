import os
import numpy as np
import cv2

#根据图像时间戳和Lidar时间戳，软同步
def softCalibration(root_dir):
    image_dir = os.path.join(root_dir,'images')
    image_list = os.listdir(image_dir)
    image_list = [x for x in image_list if x.endswith('L.png')]
    image_list.sort(key=lambda x:float(x[:-5]))
    image_timestamp = []
    lidar_path = os.path.join(root_dir, 'lidar')
    if not os.path.exists(lidar_path):
        os.makedirs(lidar_path)
    # 删除lidar_path下的所有文件
    for i in os.listdir(lidar_path):
        os.remove(os.path.join(lidar_path,i))
    for i in range(len(image_list)):
        i = image_list[i].split('L.png')[0]
        image_timestamp.append(float(i))
    for i in range(16):
        lidar_dir = os.path.join(root_dir,'seq'+str(i+1),'lidar_360')
        lidar_list = os.listdir(lidar_dir)
        lidar_list.sort(key=lambda x:float(x[:-4]))
        for j in range(len(lidar_list)):
            lidar_timestamp = lidar_list[j].split('.')[0]
            lidar_timestamp = float(lidar_timestamp)
            for k in range(len(image_timestamp)):
                if abs(lidar_timestamp - image_timestamp[k]) < 0.02:
                    # 按照image_timestamp[k]重新保存lidar数据
                    lidar = np.load(os.path.join(lidar_dir, lidar_list[j]))
                    np.save(os.path.join(lidar_path, str(image_timestamp[k])+'.npy'), lidar)
                    print("match!!!!!!!!")
                else:
                    print("don't match")

            
def main():
    root_dir = './data/Anti_UAV_data/val'
    softCalibration(root_dir)


if __name__ == '__main__':
    main()