# 鱼眼相机成像
# 将三维空间中的点投影到二维平面上
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def project():
    # 读取三维坐标
    points = np.load('./data/points.npy')
    print(points.shape)
    # 读取相机内参
    K = np.load('./data/K.npy')
    print(K)
    # 读取相机位姿
    pose = np.load('./data/pose.npy')
    print(pose)
    # 读取图像尺寸
    h, w = 1080, 1920
    # 读取畸变参数
    dist = np.load('./data/dist.npy')