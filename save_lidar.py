# 把pred、cls按照时间戳保存到csv文件中
import os
import numpy as np
import pandas as pd


frame = '17.1'
pred = [1,2,3]
cl = [1]
# 创建csv文件
output = pd.DataFrame(columns=["Sequence","Timestamp","Position","Classification"])


seq = frame.split('.')[0]
time = frame.split('.')[1]
# 保存数据
output = output.append([{'Sequence':frame, 'Timestamp':frame, 'Position':pred, 'Classification':cl}], ignore_index=True)

# 保存到csv文件
output.to_csv('output.csv', index=False)