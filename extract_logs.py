import re
import pandas as pd

# 初始化数据存储
epoch_data = []
iteration_data = []

# 读取日志文件
with open('logs.txt', 'r') as file:
    lines = file.readlines()

# 正则表达式模式
epoch_pattern = re.compile(r'Validation Results - Epoch: (\d+)')
map_pattern = re.compile(r'mAP:([\d.]+)%')
rank1_pattern = re.compile(r'CMC curve, Rank-1\s+:([\d.]+)%')
rank5_pattern = re.compile(r'CMC curve, Rank-5\s+:([\d.]+)%')
rank10_pattern = re.compile(r'CMC curve, Rank-10\s+:([\d.]+)%')
iteration_pattern = re.compile(r'Epoch\[(\d+)\] Iteration\[(\d+)/\d+\] Loss: ([\d.]+), Acc: ([\d.]+)')

# 临时变量
current_epoch = None
mAP = rank1 = rank5 = rank10 = None

# 解析日志文件
for line in lines:
    # 检查是否是epoch数据
    epoch_match = epoch_pattern.search(line)
    if epoch_match:
        current_epoch = int(epoch_match.group(1))
        continue

    map_match = map_pattern.search(line)
    if map_match:
        mAP = float(map_match.group(1))
        continue
    
    rank1_match = rank1_pattern.search(line)
    if rank1_match:
        rank1 = float(rank1_match.group(1))
        continue
    
    rank5_match = rank5_pattern.search(line)
    if rank5_match:
        rank5 = float(rank5_match.group(1))
        continue
    
    rank10_match = rank10_pattern.search(line)
    if rank10_match:
        rank10 = float(rank10_match.group(1))
        epoch_data.append([current_epoch, mAP, rank1, rank5, rank10])
        current_epoch = mAP = rank1 = rank5 = rank10 = None
        continue
    
    # 检查是否是iteration数据
    iteration_match = iteration_pattern.search(line)
    if iteration_match:
        epoch = int(iteration_match.group(1))
        iteration = int(iteration_match.group(2))
        loss = float(iteration_match.group(3))
        acc = float(iteration_match.group(4))
        iteration_data.append([epoch, iteration, loss, acc])

# 转换为DataFrame
epoch_df = pd.DataFrame(epoch_data, columns=['Epoch', 'mAP', 'Rank-1', 'Rank-5', 'Rank-10'])
iteration_df = pd.DataFrame(iteration_data, columns=['Epoch', 'Iteration', 'Loss', 'Acc'])

# 保存到Excel
with pd.ExcelWriter('log_results.xlsx') as writer:
    epoch_df.to_excel(writer, sheet_name='Epoch Data', index=False)
    iteration_df.to_excel(writer, sheet_name='Iteration Data', index=False)

print("日志数据已保存到log_results.xlsx")
