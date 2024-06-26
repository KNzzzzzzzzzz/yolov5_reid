import pandas as pd
import matplotlib.pyplot as plt

# 读取之前保存的Excel文件
epoch_df = pd.read_excel('./log_results.xlsx', sheet_name='Epoch Data')
iteration_df = pd.read_excel('./log_results.xlsx', sheet_name='Iteration Data')

# 画mAP图
plt.figure(figsize=(10, 6))
plt.plot(epoch_df['Epoch'], epoch_df['mAP'], label='mAP')
plt.xlabel('Epoch')
plt.ylabel('mAP')
plt.title('mAP')
plt.legend()
plt.savefig('./map.png')
plt.close()

# 画Rank-1, Rank-5, Rank-10的图
plt.figure(figsize=(10, 6))
plt.plot(epoch_df['Epoch'], epoch_df['Rank-1'], label='Rank-1')
plt.plot(epoch_df['Epoch'], epoch_df['Rank-5'], label='Rank-5')
plt.plot(epoch_df['Epoch'], epoch_df['Rank-10'], label='Rank-10')
plt.xlabel('Epoch')
plt.ylabel('Percentage')
plt.title('CMC Curves')
plt.legend()
plt.savefig('./rank.png')
plt.close()

# 计算每个epoch的平均Loss和Acc
avg_loss_df = iteration_df.groupby('Epoch')['Loss'].mean().reset_index()
avg_acc_df = iteration_df.groupby('Epoch')['Acc'].mean().reset_index()

# 画平均Loss图
plt.figure(figsize=(10, 6))
plt.plot(avg_loss_df['Epoch'], avg_loss_df['Loss'], label='Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Average Loss')
plt.legend()
plt.savefig('./loss.png')
plt.close()

# 画平均Acc图
plt.figure(figsize=(10, 6))
plt.plot(avg_acc_df['Epoch'], avg_acc_df['Acc'], label='Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Average Accuracy')
plt.legend()
plt.savefig('./acc.png')
plt.close()

print("图表已保存到map.png、rank.png、loss.png和acc.png")
