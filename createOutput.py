import pandas as pd
import os
import numpy as np
import scipy.interpolate as spi
import sys

output_360_path = './output_360.csv'
output_avia_path = './output_avia.csv'

def read_output(path):
    output = pd.read_csv(path)
    seq = output['Sequence']
    timestamp = output['Timestamp']
    position = output['Position']
    cl = output['Classification']
    seq = seq.values
    timestamp = timestamp.values
    position = position.values
    cl = cl.values
    return seq, timestamp, position, cl


def main():
    seq_360, timestamp_360, position_360, cl_360 = read_output(output_360_path)
    seq_avia, timestamp_avia, position_avia, cl_avia = read_output(output_avia_path)
    final_output = pd.DataFrame(columns=['Sequence', 'Timestamp', 'Position', 'Classification'])
    total_distance = 0
    total_distance_x = 0
    total_distance_y = 0
    total_distance_z = 0
    correct_dict = {}
    for i in range(len(seq_360)):
        position_360[i] = position_360[i].split(']')[0].split('tensor([')[1]
        position_avia[i] = position_avia[i].split(']')[0].split('tensor([')[1]
        if not position_360[i] == '' and not position_avia[i] == '':
            # 计算距离
            position_360[i] = position_360[i].split('[')[1].split(',')[:3]
            position_avia[i] = position_avia[i].split('[')[1].split(',')[:3]
            position_360[i] = [float(x) for x in position_360[i]]
            position_avia[i] = [float(x) for x in position_avia[i]]
            distance = np.linalg.norm(np.array(position_360[i]) - np.array(position_avia[i]))
            total_distance += distance
            x_distance = (position_360[i][0] - position_avia[i][0])
            y_distance = (position_360[i][1] - position_avia[i][1])
            z_distance = (position_360[i][2] - position_avia[i][2])
            total_distance_x += x_distance - 6.45
            total_distance_y += y_distance - (-2.703)
            total_distance_z += z_distance - (-10.7038)
            # 保存到字典里
            correct_dict[seq_360[i]] = [x_distance, y_distance, z_distance]
    avrage_distance = total_distance / len(seq_360)
    avrage_distance_x = total_distance_x / len(seq_360)
    avrage_distance_y = total_distance_y / len(seq_360)
    avrage_distance_z = total_distance_z / len(seq_360)
    print('avrage_distance:', avrage_distance)
    print('avrage_distance_x:', avrage_distance_x)
    print('avrage_distance_y:', avrage_distance_y)
    print('avrage_distance_z:', avrage_distance_z)

if __name__ == '__main__':
    main()