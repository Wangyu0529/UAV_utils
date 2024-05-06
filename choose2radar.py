import os
import numpy as np


def choose2radar(root_path):
    root_list = os.listdir(root_path)
    for i in range(len(root_list)):
        if i > 1:
            seq_path = os.path.join(root_path, root_list[i])
            radar_path = os.path.join(seq_path, 'radar')
            radar_list = os.listdir(radar_path)
            radar_list.sort(key=lambda x:float(x[:-4]))
            chosen_radar = []
            for j in range(len(radar_list)):
                radar = np.load(os.path.join(radar_path, radar_list[j]))
                #radar数据量大于0
                if radar.size > 0:
                    chosen_radar.append(radar_list[j])
            print(len(chosen_radar))
    return chosen_radar


def main():
    root_path = './data/Anti_UAV_data/train'
    choose2radar(root_path)
    return 0

if __name__ == '__main__':
    main()