import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
 
 
def point_show(path):
    #读取npy文件
    point_cloud = np.load(npy_path)
    #读入点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    
    # 路径、输入格式、删除包含NAN的所有点、删除包含无限值的所有点、可视化进度条
    print(pcd)  # 输出点云点的个数
    # 点云采用多种方式显示

    # pcd.paint_uniform_color([0, 1, 1])  # 固定颜色显示
    # pcd.colors = o3d.utility.Vector3dVector(np.random.uniform(0, 1, (1,3)))  # 随机颜色显示
    o3d.visualization.draw_geometries([pcd], window_name='Point Cloud View', width=1920, height=1080, left=50, top=50,
                                      point_show_normal=True, mesh_show_wireframe=True, mesh_show_back_face=True)
    # zoom=0.3412,front=[0.4257, -0.2125, -0.8795],lookat=[2.6172, 2.0475, 1.532],up=[-0.0694, -0.9768, 0.2024]
    # 显示内容、窗口标题、长、宽、左边距、右边距、是否可视化法线、是否可视化网络线框、是否可视化网络三角形背面
    # plt点云数据
    fig = plt.figure(dpi=120)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('Point Cloud')
    ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:,2] ,s=0.1, marker='*', cmap='jet')
    ax.axis()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

   
 
 
if __name__ == "__main__":
    npy_path = 'code/1706256744.600499.npy'
    point_show(npy_path)