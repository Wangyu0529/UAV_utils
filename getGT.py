# TODO 先根据groundtruth的三维值进行球面投影，然后再二维投影到图像上
# TODO 根据二维坐标画出框，保存到txt文件中
import numpy as np
import cv2
import os


# 内参
D_left = np.array([-0.35828593,  0.34304531, -0.00007972, -0.00060307])
D_right = np.array([-0.18785813,  1.07267765, -0.00011855, -0.00021921])
camera_matrix_left = np.array([[1844.9864629191054/2.9414247486244967, 0.00000000e+00, 615.0248705672041],
                            [0.00000000e+00, 1845.5803330648505/2.9414247486244967, 507.23461275157297],
                            [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
     
camera_matrix_right = np.array([[2106.85398494/3.436494190606926, 0.00000000e+00, 665.83488343],
                            [0.00000000e+00, 2106.8526832/3.436494190606926, 5.09768819e+02],
                            [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

alpha = -0.03
# 先归一化，再投影
def project(gt, is_left=True):
    x,y,z = gt[0],gt[1],gt[2]
    # 归一化
    d = np.sqrt(np.sum(gt**2))
    a = x/d
    b = y/d
    c = z/d
    r = np.sqrt(a**2 + b**2)
    
    theta = np.arctan2(r,abs(c))
    # 畸变矫正
    if is_left:
        D = D_left
        theta_d = theta*(1+D[0]*theta**2+D[1]*theta**4+D[2]*theta**6+D[3]*theta**8)
        # 投影
        x_d = theta_d/r*a
        y_d = theta_d/r*b
        # 像素坐标
        u = (camera_matrix_left[0,0]*(x_d+alpha*y_d) + camera_matrix_left[0,2])
        v = (camera_matrix_left[1,1]*y_d + camera_matrix_left[1,2])
    else:
        D = D_right
        theta_d = theta*(1+D[0]*theta**2+D[1]*theta**4+D[2]*theta**6+D[3]*theta**8)
        # 投影
        x_d = theta_d*a/r
        y_d = theta_d*b/r
        # 像素坐标
        u = camera_matrix_right[0,0]*(x_d+alpha*y_d) + camera_matrix_right[0,2]
        v = camera_matrix_right[1,1]*y_d + camera_matrix_right[1,2]
    return u,v,d

def InversePM(center):
    # 根据鱼眼相机几何关系，计算角度
    u = center[0]
    v = center[1]
    y_d = (v - camera_matrix_left[1,2])/camera_matrix_left[1,1]
    x_d = (u - camera_matrix_left[0,2])/camera_matrix_left[0,0] - alpha*y_d
    theta_d = np.sqrt(x_d**2 + y_d**2)
    a = x_d/theta_d
    b = y_d/theta_d
    r = 1
    c = 1/np.tan(theta_d)
    # 坐标
    target = np.array([a,b,c]) # x,y,z
    return target




def main():
    root_dir = './data/Anti_UAV_data/train'
    for i in range(len(os.listdir(root_dir))):
        seq_path =  os.path.join(root_dir,'seq'+str(i+1))
        print(seq_path)
        gt_path = os.path.join(seq_path, 'gt')
        gt_list = os.listdir(gt_path)
        gt_list.sort(key=lambda x:float(x[:-4]))
        image_path = os.path.join(root_dir, 'images')
        cls_path = os.path.join(seq_path, 'cls')
        cv_list = os.listdir(image_path)       
        cv_list.sort(key=lambda x:float(x[:-5]))
        cv_list = [x.replace('L.png','.png') for x in cv_list if x.endswith('L.png')]
        offsetL_path = os.path.join(root_dir, 'labels')
        offsetR_path = os.path.join(root_dir, 'labels')
        visual_path = os.path.join(root_dir, 'visual')
        if not os.path.exists(offsetL_path):
            os.makedirs(offsetL_path)
        if not os.path.exists(offsetR_path):
            os.makedirs(offsetR_path)
        for j in range(len(gt_list)):
            if gt_list[j].replace('.npy', '.png') in cv_list:
                gt_dir = os.path.join(gt_path, gt_list[j])
                gt = np.load(gt_dir)
                ul,vl,d = project(gt)
                ur,vr,_ = project(gt, is_left=False)
                D = np.array([0.33463228, -0.95258994,  1.51580406])
                offset_x = int(120*np.sqrt(np.sum(D**2))/d)
                offset_y = int(80*np.sqrt(np.sum(D**2))/d)
                cls = np.load(os.path.join(cls_path, gt_list[j]))
                # 按照yolo格式保存框
                with open(os.path.join(offsetL_path, gt_list[j].replace('.npy','L.txt')),'w') as f:
                    f.write('%f %f %f %f %f\n'%(int(cls), int(ul)/1280, int(vl)/960, offset_x/1280, offset_y/960))
                f.close()
                with open(os.path.join(offsetR_path, gt_list[j].replace('.npy','R.txt')),'w') as f:
                    f.write('%f %f %f %f %f\n'%(int(cls), int(ur)/1280, int(vr)/960, offset_x/1280, offset_y/960))
                f.close()
                # 画框
                img = cv2.imread(os.path.join(image_path, gt_list[j].replace('.npy', '.png')))
                cv2.rectangle(img, (int(ul)-offset_x, int(vl)-offset_y), (int(ul)+offset_x, int(vl)+offset_y), (0,0,255), 2)
                cv2.rectangle(img, (int(ur)-offset_x, int(vr)-offset_y), (int(ur)+offset_x, int(vr)+offset_y), (0,0,255), 2)
                cv2.imwrite(os.path.join(visual_path, gt_list[j].replace('.npy', '.png')), img)
            


    

if __name__ == '__main__':
    main()