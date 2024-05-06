import os
import numpy as np
import cv2

if __name__ == '__main__':
    root_dir = './data/Anti_UAV_data/train'
    image_path = os.path.join(root_dir, 'image')
    image_list = os.listdir(image_path)
    image_list.sort(key=lambda x:float(x[:-4]))
    for i in range(len(image_list)):
        # 如果img不以L.png或R.png结尾，则分割成左右两张图
        if not image_list[i].endswith('L.png') and not image_list[i].endswith('R.png'):
            img = cv2.imread(os.path.join(image_path, image_list[i]))
            cv2.imwrite(os.path.join(image_path, image_list[i].replace('.png','L.png')), img[:,:1280])
            cv2.imwrite(os.path.join(image_path, image_list[i].replace('.png','R.png')), img[:,1280:])
        # 删除原图
            os.remove(os.path.join(image_path, image_list[i]))      