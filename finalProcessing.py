import pandas as pd
import os
import numpy as np
import scipy.interpolate as spi


# 读取test_timestamp.csv,获得前两列seq,timestamp_360
def read_test_timestamp():
    test_timestamp = pd.read_csv('./test_timestamp.csv')
    seq_360 = test_timestamp['Sequence']
    timestamp_360 = test_timestamp['Timestamp']
    return seq_360, timestamp_360

output_360_path = './output_360.csv'
output_avia_path = './output_avia.csv'





def read_output(path):
    output = pd.read_csv(path)
    seq_360 = output['Sequence']
    timestamp_360 = output['Timestamp']
    position_360 = output['Position']
    cl_360 = output['Classification']
    seq_360 = seq_360.values
    timestamp_360 = timestamp_360.values
    position_360 = position_360.values
    cl_360 = cl_360.values
    return seq_360, timestamp_360, position_360, cl_360
# 修正误差
correct_table = [6.45, -2.703, -10.7038]
total_correct = [0.6589986339018487,-9.220989933866518,26]
def main():
    # 读取test_timestamp.csv,获得前两列seq,timestamp,建立字典
    seq_dict = {}
    Seq, Timestamp = read_test_timestamp()
    Seq = Seq.values
    Timestamp = Timestamp.values
    # 把所有值舍去小数点后面的值
    index_dict = {}
    for i in range(len(Seq)):
        new_Timestamp = Timestamp[i] - Timestamp[i]%1
        seq_dict[Seq[i]] = int(new_Timestamp)
    # 按值排序
    seq_dict = dict(sorted(seq_dict.items(), key=lambda item: item[1]))
    for i, key in enumerate(seq_dict.keys()):
        index_dict[i] = seq_dict[key]
    print(index_dict)
    seq_360, timestamp_360, position_360, cl_360 = read_output(output_360_path)
    seq_avia, timestamp_avia, position_avia, cl_avia = read_output(output_avia_path)
    final_output = pd.DataFrame(columns=['Sequence', 'Timestamp', 'Position', 'Classification'])
    for i in range(len(seq_360)):
        position_360[i] = position_360[i].split(']')[0].split('tensor([')[1]
        position_avia[i] = position_avia[i].split(']')[0].split('tensor([')[1]
        if (position_360[i]==''):
            if (position_avia[i]==''):
                position_360[i] = total_correct
            else:
                position_360[i] = position_avia[i].split('[')[1].split(',')[:3]
                # 修正数值
                position_360[i] = [float(position_360[i][0])-correct_table[0], float(position_360[i][1])-correct_table[1], float(position_360[i][2])-correct_table[2]]       
        else:
            position_360[i] = position_360[i].split('[')[1].split(',')[:3]
        cl_360[i] = cl_360[i].split('[')[1].split(']')[0].split(',')[0]
        if (cl_360[i]==''):
            cl_360[i] = 2
    k = 0
    now_seq = index_dict[k]
    for i in range(len(seq_360)-1):
        if seq_360[i] <= now_seq:
            if position_360[i] == total_correct and position_360[i+1] != total_correct and position_360[i+2] != total_correct:
                for j in range(i+1):
                    if position_360[j] == total_correct:
                        position_360[j] = position_360[i+1]
        else:
            k += 1
            print(k)           
            now_seq = index_dict[k]
    for i in range(len(position_360)):
        position_360[i] = [float(x) for x in position_360[i]]
        # if position_360[i] == [0.0,0.0,0.0]:
        #     position_360[i] = position_360[i-1]
        cl_360[i] = float(cl_360[i])
    # 保存position_360
    position = pd.DataFrame(columns=['Sequence','Position'])
    for i in range(len(seq_360)):
        position = position.append({'Sequence': seq_360[i], 'Position': position_360[i]}, ignore_index=True)
    position.to_csv('./position_360.csv', index=False)
    

    # 把seq,timestamp按字符串拼接起来
    seq_timestamp = []
    for i in range(len(seq_360)):
        seq_timestamp.append(str(seq_360[i]) + '.' + str(timestamp_360[i]))
    # 按Seq将数据分组
    for j in range(9):
        test_timestamp = []
        test_time = []
        test_position = []
        test_cl_360 = [] 
        for i in range(len(Seq)):
            if Seq[i] == 'seq000'+str(j+1):
                test_time.append(int(Timestamp[i]))
                test_timestamp.append(Timestamp[i])
        time_list = [] 
        position_list = []
        cl_360_list = []
        for i in range(len(seq_timestamp)):
            if seq_360[i] in test_time:
                time_list.append(float(seq_timestamp[i]))
                position_360[i] = [float(x) for x in position_360[i]]
                position_list.append(position_360[i])
                cl_360_list.append(cl_360[i])
        # 对position,cl_360进行np插值
        position_list = np.array(position_list)
        for i in range(3):
            position_list[:,i] = spi.interp1d(time_list, position_list[:,i], kind='quadratic')(time_list)
        cl_360_list = np.array(cl_360_list)
        new_position = np.zeros((len(test_timestamp),3))
        new_cl_360 = np.zeros(len(test_timestamp))
        for i in range(3):
            new_position[:,i] = np.interp(test_timestamp, time_list, position_list[:,i])
        print(new_position.shape)
        new_cl_360 = np.interp(test_timestamp, time_list, cl_360_list)
        # 将插值后的数据写入文件
        for i in range(len(test_timestamp)):
            final_output = final_output.append({'Sequence': 'seq000'+str(j+1), 'Timestamp': test_timestamp[i], 'Position': [new_position[i,0],new_position[i,1],new_position[i,2]], 'Classification': int(new_cl_360[i])}, ignore_index=True)
    for j in range(50):
        test_timestamp = []
        test_time = []
        test_position = []
        test_cl_360 = [] 
        for i in range(len(Seq)):
            if Seq[i] == 'seq00'+str(j+10):
                test_time.append(int(Timestamp[i]))
                test_timestamp.append(Timestamp[i])
        time_list = [] 
        position_list = []
        cl_360_list = []
        for i in range(len(seq_timestamp)):
            if seq_360[i] in test_time:
                time_list.append(float(seq_timestamp[i]))
                position_360[i] = [float(x) for x in position_360[i]]
                position_list.append(position_360[i])
                cl_360_list.append(cl_360[i])
        # 对position,cl_360进行np插值
        position_list = np.array(position_list)
        for i in range(3):
            position_list[:,i] = spi.interp1d(time_list, position_list[:,i], kind='quadratic')(time_list)
        cl_360_list = np.array(cl_360_list)
        new_position = np.zeros((len(test_timestamp),3))
        new_cl_360 = np.zeros(len(test_timestamp))
        for i in range(3):
            new_position[:,i] = np.interp(test_timestamp, time_list, position_list[:,i])
        print(new_position.shape)
        new_cl_360 = np.interp(test_timestamp, time_list, cl_360_list)
        # 将插值后的数据写入文件
        for i in range(len(test_timestamp)):
            final_output = final_output.append({'Sequence': 'seq00'+str(j+10), 'Timestamp': test_timestamp[i], 'Position': [new_position[i,0],new_position[i,1],new_position[i,2]], 'Classification': int(new_cl_360[i])}, ignore_index=True)
    final_output.to_csv('./mmaud_results.csv', index=False)
    


if __name__=='__main__':
    main()