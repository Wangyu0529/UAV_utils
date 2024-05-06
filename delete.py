import os
import numpy as np
import cv2

def delete(root_dir):
    image_path = os.path.join(root_dir, 'images')
    image_list = os.listdir(image_path)
    image_list.sort(key=lambda x:float(x[:-5]))
    label_path = os.path.join(root_dir, 'labels')
    label_list = os.listdir(label_path)
    label_list.sort(key=lambda x:float(x[:-5]))
    for i in range(len(label_list)):
        with open(os.path.join(label_path, label_list[i]), 'r') as f:
            lines = f.readlines()
            lines = [x.strip() for x in lines]
            for x in lines:
                lines = x.split(' ')
            lines = [float(y) for y in lines]
            print(lines)
            if lines[1]<0 or lines[1]>1 or lines[2]<0 or lines[2]>1 or lines[3]<0 or lines[3]>1 or lines[4]<0 or lines[4]>1:
                os.remove(os.path.join(label_path, label_list[i]))
                os.remove(os.path.join(image_path, label_list[i].replace('.txt','.png')))

def delete_all(root_dir):
    image_path = os.path.join(root_dir, 'images')
    label_path = os.path.join(root_dir, 'labels')
    # 删除文件夹下所有文件
    for i in os.listdir(image_path):
        os.remove(os.path.join(image_path, i))
    for i in os.listdir(label_path):
        os.remove(os.path.join(label_path, i))

def delete_image(root_dir):
    image_path = os.path.join(root_dir,'images')
    delete_list = [1706255586,1706255587,1706255588,1706255589,1706255590,1706255591,1706255592,1706255593,1706255594,1706255595,1706255596,1706255597,
                   1706255995,1706255996,1706255997,1706255998,1706255999,1706256000,1706256001,1706256002,1706256003,1706256004,1706256005,1706256006,1706256007,1706256008,
                   1706256258,1706256259,
                   1706256258,170625629,
                   1706256508,1706256509,1706256510,1706256511,1706256512,1706256513,1706256514,
                   1706256768,1706256769,1706256770,1706256771,1706256772,1706256773,
                   1706256805,1706256806,1706256807,1706256808,1706256809,1706256810,1706256811,1706256812,
                   1706257054,
                   1706257058,1706257059,1706257060,1706257061,1706257062,1706257063,1706257064,1706257065,1706257066,1706257067,1706257068,
                   1706257362,1706257363,1706257368,1706257369,1706257370,1706257371,
                   1706257390,1706257391,1706257392,1706257393,1706257394,1706257395,1706257396,1706257397,1706257398,1706257399,1706257400,
                   1706257428,1706257429,1706257430,1706257431,
                   1706257490,1706257491,
                   1706257627,1706257628,1706257629,1706257630,1706257631,1706257632,1706257633,1706257634,1706257638,1706257639,1706257640,
                   1706258584,1706258585,1706258586,1706258587,1706258588,1706258589,1706258590,1706258591,
                   1706258665,1706258666,1706258667,1706258668,1706258669,1706258670,1706258671,
                   1706258696,1706258687,1706258698,1706258699,
                   1706258706,1706258707,1706258708,1706258709,
                   1706258726,1706258727,1706258728,1706258729,1706258730,1706258731,1706258732,1706258733,1706258734,1706258735]
    for i in os.listdir(image_path):
        if int(i[:-12]) in delete_list:
            os.remove(os.path.join(image_path, i))
        

def main():
    root_dir = './data/Anti_UAV_data/train'
    delete_all(root_dir)

if __name__ == '__main__':
    main()
